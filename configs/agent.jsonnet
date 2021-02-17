{
  dataset_reader: {
    type: 'srl_reader.SRLDatasetReader',
    lazy: false
  },
  train_data_path: "data/AGENT_train.csv",
  validation_data_path: "data/AGENT_dev.csv",
  "test_data_path": "data/AGENT_test.csv",
  model: {
    type: 'model.MyModel',
    embedder: {},
    encoder: {}
  },
  data_loader: {
    batch_size: 10,
    shuffle: true
  },
  trainer: {}
}