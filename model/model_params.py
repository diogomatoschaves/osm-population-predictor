

feature_mapping_params = {
    "order": 2,
    "only_self_terms": True
}


regressor_name = "Lasso"

regressor_params = {
    "RandomForestRegressor": {
        "n_estimators": 200
    },
    "GradientBoostingRegressor": {
        "n_estimators": 200,
        "max_features": "sqrt",
        "max_depth": 5,
        "min_samples_split": 5
    },
    "Ridge": {},
    "Lasso": {
        "alpha": 2,
        "max_iter": 2000
    }
}

not_features = [
    'id',
    'updated',
    'ADMIN',
    'ISO_A3',
    'building_count',
    'count',
    'osm_users',
    'geometry',
    'gdp',
    'population',
    'area_km2',
    'highway_sum',
    'highway_length',
    'osm_objects'
]