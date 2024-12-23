from datetime import datetime, timedelta
from typing import Dict, List, Optional
from feast import (
    Entity,
    Feature,
    FeatureService,
    FeatureView,
    FileSource,
    ValueType,
    Field,
    Project
)
from feast.types import Float32, Float64, Int64

# Define a project
project: Project = Project(
    name="retail_prediction",
    description="A feature repository for retail prediction"
)

# Define the data source
transaction_source: FileSource = FileSource(
    name="transaction_stats_source",
    path="data/train_data.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define entities
customer: Entity = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="customer identifier",
)

store: Entity = Entity(
    name="store_id",
    join_keys=["store_id"],
    description="store identifier",
)

# Define feature view
transaction_stats: FeatureView = FeatureView(
    name="transaction_stats",
    entities=[customer, store],
    ttl=timedelta(days=3),
    schema=[
        Field(name="transaction_amount", dtype=Float32),
        Field(name="num_items", dtype=Int64),
        Field(name="seasonal_factor", dtype=Float32),
    ],
    online=True,
    source=transaction_source,
    tags={"team": "ml_team"},
)

# Define feature service
prediction_service: FeatureService = FeatureService(
    name="prediction_service",
    features=[transaction_stats],
) 