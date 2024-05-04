import os

from flask import Flask, request, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD
from recommender import recommender, calc_metric
from utils import write_embeddings, read_embeddings

app = Flask(__name__)


@app.route('/get_user_recommendations', methods=['POST'])
def get_user_recommendations():
    data = request.get_json()
    reviews = pd.DataFrame(data['reviews'])
    assets = pd.DataFrame(data['assets'])
    # Вычисление рекомендаций
    recommendations = calculate_user_recommendations(reviews, assets)
    return jsonify(recommendations)


@app.route('/get_asset_recommendations', methods=['POST'])
def get_asset_recommendations():
    data = request.get_json()
    assets = pd.DataFrame(data['assets'])
    # Вычисление рекомендаций
    recommendations = calculate_asset_recommendations(assets)
    return jsonify(recommendations)


def calculate_user_recommendations(reviews, assets):
    df_reviews = reviews
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df_reviews[['author', 'asset', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    # trainset, testset = train_test_split(dataset, test_size=0.1)
    model = SVD()
    model.fit(trainset)
    unique_author_ids = set(df_reviews['author'].unique())
    recommendations_list = []

    for author_id in unique_author_ids:
        df_test = assets[assets['author'] == author_id].copy()
        predictions_test = []
        for index, row in df_test.iterrows():
            pred = model.predict(row['author'], row['asset'])
            predictions_test.append(pred.est)

        df_test.loc[:, 'predicted_rating'] = predictions_test
        df_test = df_test.sort_values('predicted_rating', ascending=False)
        top_5_recommendations = df_test.head(5)

        for index, row in top_5_recommendations.iterrows():
            recommendation = {
                'author_id': row['author'],
                'asset_id': row['asset'],
                'predicted_rating': row['predicted_rating']
            }
            recommendations_list.append(recommendation)
    return recommendations_list


def calculate_asset_recommendations(assets):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    file_path = os.path.join(data_dir, f"embeddings_all.h5")
    if os.path.exists(file_path):
        os.remove(file_path)
    for _, asset_data in assets.iterrows():
        write_embeddings(recommender, asset_data)
    seen_pairs = set()
    distances = []
    for _, asset_data in assets.iterrows():
        asset_id = asset_data['id']
        emb_i = read_embeddings(asset_id)
        for _, asst_data in assets.iterrows():
            asst_id = asst_data['id']
            if asst_id != asset_id:
                pair = tuple(sorted([asset_id, asst_id]))
                if pair not in seen_pairs:
                    emb_j = read_embeddings(asst_id)
                    distance = calc_metric(emb_i, emb_j)
                    distances.append((asset_id, asst_id, distance))
                    seen_pairs.add(pair)

    distances_df = pd.DataFrame(distances, columns=['asset1_id', 'asset2_id', 'distance'])
    distances_dict = distances_df.to_dict('records')
    return distances_dict


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
