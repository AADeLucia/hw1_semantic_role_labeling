{
  dataset_reader: {
    type: "srl_reader.SRLDatasetReader",
    lazy: false
  },
  train_data_path: 'data/AGENT_train.txt',
  model: {},
  data_loader: {
    batch_sampler: {
      "type": "bucket",
      "batch_size": 10
    },
  },
  trainer: {}
}