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
  "train_data_path": "data/AGENT_train.csv",
  "validation_data_path": "data/AGENT_dev.csv",
  "test_data_path": "data/AGENT_test.csv",
  "model": {
    "type": "model.MyModel",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
          "embedding_dim": 50,
          "trainable": false
        }
      }
    },
    "encoder": {
      "type": 'lstm',
      "input_size": 50,
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
    "num_epochs": 1,
    "cuda_device": -1,
    "validation_metric": "-loss",
    "optimizer": {
      "type": "adam",
      "lr": 0.003
    }
  }
}