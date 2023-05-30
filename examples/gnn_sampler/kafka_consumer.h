/** Copyright 2020 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef EXAMPLES_GNN_SAMPLER_KAFKA_CONSUMER_H_
#define EXAMPLES_GNN_SAMPLER_KAFKA_CONSUMER_H_

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <librdkafka/rdkafka.h>
#include <librdkafka/rdkafkacpp.h>

#include "util.h"

/** Kafka consumer class
 *
 * A kafka consumer class based on librdkafka, can be to get streaming
 * data from one kafka topic.
 */
class KafkaConsumer {
 public:
  KafkaConsumer(int worker_id, const std::string& broker_list,
                const std::string& group_id, const std::string& topic,
                int partition_num, int64_t time_interval, int batch_size)
      : worker_id_(worker_id),
        topic_(topic),
        partition_num_(partition_num),
        time_interval_ms_(time_interval * 1000) {
    CHECK_GT(partition_num, 0);
    CHECK_GT(batch_size, 0);
    CHECK_GT(time_interval, 0);
    batch_size_per_partition_ = batch_size / partition_num;

    RdKafka::Conf* conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    std::string rdkafka_err;
    if (conf->set("metadata.broker.list", broker_list, rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set metadata.broker.list: " << rdkafka_err;
    }
    if (conf->set("group.id", group_id, rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set group.id: " << rdkafka_err;
    }
    if (conf->set("enable.auto.commit", "false", rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set enable.auto.commit: " << rdkafka_err;
    }
    if (conf->set("auto.offset.reset", "earliest", rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set auto.offset.reset: " << rdkafka_err;
    }

    qmq_.resize(partition_num);
    emq_.resize(partition_num);
    for (int i = 0; i < partition_num; ++i) {
      consumer_ptrs_[i] = std::shared_ptr<RdKafka::KafkaConsumer>(
          RdKafka::KafkaConsumer::create(conf, rdkafka_err));
      if (!consumer_ptrs_[i]) {
        LOG(ERROR) << "Failed to create rdkafka consumer for partition "
                   << rdkafka_err;
      }
      std::vector<RdKafka::TopicPartition*> topic_partitions;
      RdKafka::TopicPartition* topic_partition =
          RdKafka::TopicPartition::create(topic_, i);
      consumer_ptrs_[i]->assign({topic_partition});
      delete topic_partition;
      topic_partition = nullptr;
      consumer_ptrs_[i]->subscribe({topic_});

      qmq_[i] =
          std::make_shared<grape::BlockingQueue<std::vector<std::string>>>();
      emq_[i] =
          std::make_shared<grape::BlockingQueue<std::vector<std::string>>>();
      qmq_[i]->SetLimit(16);
      emq_[i]->SetLimit(16);
      qmq_[i]->SetProducerNum(1);
      emq_[i]->SetProducerNum(1);
    }
    delete conf;  // release the memory resource

    startFetch();
  }

  ~KafkaConsumer() = default;

  /** Fetch stream data from the topic given when init
   *  at the specified offset.
   */
  void ConsumeMessages(std::vector<std::string>& query_messages,
                       std::vector<std::string>& edge_messages) {
    for (int i = 0; i < partition_num_; ++i) {
      std::vector<std::string> qs, es;
      qmq_[i]->Get(qs);
      emq_[i]->Get(es);
      std::copy(qs.begin(), qs.end(), std::back_inserter(query_messages));
      std::copy(es.begin(), es.end(), std::back_inserter(edge_messages));
    }
    if (!query_messages.empty() || !edge_messages.empty()) {
      LOG(INFO) << "consumed " << query_messages.size() << " query messages, "
                << edge_messages.size() << " edge messages.";
    }
  }

  inline std::string topic() { return topic_; }

 private:
  void startFetch() {
    for (int i = 0; i < partition_num_; ++i) {
      std::thread t = std::thread([&, i] {
        while (true) {
          std::vector<std::string> qs, es;
          fetchBatch(i, qs, es);
          qmq_[i]->Put(std::move(qs));
          emq_[i]->Put(std::move(es));
        }
      });
      t.detach();
      VLOG(1) << "[proc" << worker_id_ << "] start fetch thread on partition "
              << i;
    }
  }

  void fetchBatch(int partition, std::vector<std::string>& query_messages,
                  std::vector<std::string>& edge_messages) {
    edge_messages.reserve(batch_size_per_partition_);
    // Create a consumer dispatcher
    auto consumer_ptr_ = consumer_ptrs_[partition];

    int msg_cnt = 0;
    int64_t first_msg_ts = 0, cur_msg_ts = 0;

    int msg_len;
    const char* msg_payload;
    int64_t timestamp;
    std::string msg_data;

    auto process = [&](RdKafka::Message* message) -> bool {
      switch (message->err()) {
      case RdKafka::ERR__TIMED_OUT:
        return true;

      case RdKafka::ERR_NO_ERROR:
        /* process message */
        msg_len = message->len();
        msg_payload = static_cast<char*>(message->payload());
        timestamp = message->timestamp().timestamp;

        msg_data = std::string(msg_payload, msg_len);
        if (!msg_data.empty()) {
          if (msg_data[0] != '#') {
            if (msg_data[0] == 'e') {
              edge_messages.push_back(msg_data.substr(2, msg_data.size() - 1));
            } else if (msg_data[0] == 'q') {
              query_messages.push_back(msg_data.substr(2, msg_data.size() - 1));
            }
            ++msg_cnt;
          }
        }
        if (!first_msg_ts) {
          first_msg_ts = timestamp;
        }
        cur_msg_ts = timestamp;
        if (msg_cnt >= this->batch_size_per_partition_ ||
            cur_msg_ts - first_msg_ts > this->time_interval_ms_) {
          msg_cnt = 0;
          first_msg_ts = 0;
          return false;
        } else {
          return true;
        }

      case RdKafka::ERR__PARTITION_EOF:
        LOG(ERROR) << "Reached EOF on partition";
        return false;

      case RdKafka::ERR__UNKNOWN_TOPIC:
      case RdKafka::ERR__UNKNOWN_PARTITION:
        LOG(ERROR) << "Topic or partition error: " << message->errstr();
        return false;

      default:
        LOG(ERROR) << "Unhandled kafka error: " << message->errstr();
        return false;
      }
    };

    while (true) {
      RdKafka::Message* message = consumer_ptr_->consume(1000 * 60);  // 60s
      bool cont = process(message);
      delete message;
      if (!cont) {
        break;
      }
    }
  }

  template <typename T>
  using mq_t = std::shared_ptr<grape::BlockingQueue<std::vector<T>>>;
  std::vector<mq_t<std::string>> qmq_;
  std::vector<mq_t<std::string>> emq_;

  int worker_id_;

  std::string topic_;
  int partition_num_;
  int64_t time_interval_ms_;
  int batch_size_per_partition_;

  std::map<int, std::shared_ptr<RdKafka::KafkaConsumer>> consumer_ptrs_;
};

#endif  // EXAMPLES_GNN_SAMPLER_KAFKA_CONSUMER_H_
