{
  "dataset_reader": {
    "type": "srl_reader.SRLDatasetReader",
    "lazy": false,
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      }
    }
  },
  "train_data_path": std.extVar("SRL_TRAIN"),
  "validation_data_path": std.extVar("SRL_DEV"),
  "test_data_path": std.extVar("SRL_TEST"),
  "model": {
    "type": "model.MyModel",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "/Users/alexandra/embeddings/glove.840B.300d.zip",
          "embedding_dim": 300,
          "trainable": false
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 25,
      "bidirectional": true
    }
  },
  "data_loader": {
    "batch_size": 10,
    "shuffle": true
  },
  "vocabulary": {
    "only_include_pretrained_words": false
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device": -1,
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adam",
      "lr": 0.003
    }
  }
}