

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
    },
    "Ridge": {},
    "Lasso": {
        "alpha": 2,
        "max_iter": 2000
    }
}

grid_search = False

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
delete_outliers_bool = True
