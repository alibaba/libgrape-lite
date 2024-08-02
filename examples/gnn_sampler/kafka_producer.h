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

#ifndef EXAMPLES_GNN_SAMPLER_KAFKA_PRODUCER_H_
#define EXAMPLES_GNN_SAMPLER_KAFKA_PRODUCER_H_

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include <librdkafka/rdkafka.h>
#include <librdkafka/rdkafkacpp.h>

/** Kafka producer class
 *
 * A kafka producer class based on librdkafka, can be used to produce
 * stream data to one topic.
 */
class KafkaProducer {
 public:
  KafkaProducer() = default;

  explicit KafkaProducer(const std::string& broker_list,
                         const std::string& topic)
      : brokers_(broker_list), topic_(topic) {
    RdKafka::Conf* conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    std::string rdkafka_err;
    if (conf->set("metadata.broker.list", broker_list, rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set metadata.broker.list: " << rdkafka_err;
    }
    if (conf->set("metadata.broker.list", broker_list, rdkafka_err) !=
        RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set metadata.broker.list: " << rdkafka_err;
    }
    // for producer's internal queue.
    if (conf->set("queue.buffering.max.messages",
                  std::to_string(internal_buffer_size_),
                  rdkafka_err) != RdKafka::Conf::CONF_OK) {
      LOG(WARNING) << "Failed to set queue.buffering.max.messages: "
                   << rdkafka_err;
    }

    producer_ = std::unique_ptr<RdKafka::Producer>(
        RdKafka::Producer::create(conf, rdkafka_err));
    if (!producer_) {
      LOG(ERROR) << "Failed to create kafka producer: " << rdkafka_err;
    }
    delete conf;  // release the memory resource
  }

  ~KafkaProducer() = default;

  void AddMessage(const std::string& message) {
    if (message.empty()) {
      return;
    }
    RdKafka::ErrorCode err = producer_->produce(
        topic_, RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
        static_cast<void*>(const_cast<char*>(message.c_str())) /* value */,
        message.size() /* size */, NULL, 0, 0 /* timestamp */,
        NULL /* delivery report */);
    if (err != RdKafka::ERR_NO_ERROR) {
      LOG(ERROR) << "Failed to output to kafka: " << RdKafka::err2str(err);
    }
    pending_count_ += 1;
    if (pending_count_ == 1024 * 128) {
      producer_->flush(1000 * 60);  // 60s
      pending_count_ = 0;
    }
  }

  inline std::string topic() { return topic_; }

 private:
  static const constexpr int internal_buffer_size_ = 1024 * 1024;

  size_t pending_count_ = 0;
  std::string brokers_;
  std::string topic_;
  std::unique_ptr<RdKafka::Producer> producer_;
};

/**
 * A kafka output stream that can be used to flush messages into kafka prodcuer.
 */
class KafkaOutputStream : public std::ostream {
 public:
  explicit KafkaOutputStream(std::shared_ptr<KafkaProducer> producer)
      : std::ostream(new KafkaBuffer(producer)) {}

  ~KafkaOutputStream() { delete rdbuf(); }

  void close() {}

 private:
  class KafkaBuffer : public std::stringbuf {
   public:
    explicit KafkaBuffer(std::shared_ptr<KafkaProducer>& prodcuer)
        : producer_(prodcuer) {}

    int sync() override {
      producer_->AddMessage(this->str());
      this->str("");
      return 0;
    }

   private:
    std::shared_ptr<KafkaProducer> producer_;
  };
};

#endif  // EXAMPLES_GNN_SAMPLER_KAFKA_PRODUCER_H_
