def build_inventory(data_root: str, out_path: str) -> None:
    """
    Scans a tpugraphs-style directory (npz/layout/... and npz/tile/...)
    and writes a parquet/csv inventory with:
      - collection, split, file_path
      - bytes_on_disk
      - n_nodes, n_edges, n_configs
      - schema_hash (keys+dtypes)
      - flags: has_nan, has_inf, edge_oob, runtime_oob
    """
