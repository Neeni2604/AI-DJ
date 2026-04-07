from fma_db.ingest import import_fma_dataset

result = import_fma_dataset(
    db_path="fma_db/data/fma.db",
    metadata_dir="fma_db/data/fma_metadata",
    audio_dir="fma_db/data/fma_small",
    subset_name="small"
)
print(result)