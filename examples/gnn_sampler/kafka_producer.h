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

#include <boost/algorithm/string.hpp>
#include <boost/iostreams/categories.hpp>

#include "cppkafka/configuration.h"
#include "cppkafka/cppkafka.h"
#include "cppkafka/utils/buffered_producer.h"

using cppkafka::BufferedProducer;
using cppkafka::Configuration;
using cppkafka::MessageBuilder;

/** Kafka producer class
 *
 * A stream-data-push class based on cppkafka (a cpp wraper of librdkafka).
 * You can use this class to produce stream data to one topic.
 */
class KafkaProducer {
 public:
  KafkaProducer() = default;

  explicit KafkaProducer(const std::string& brokers, const std::string& topic)
      : brokers_(brokers), topic_(topic) {
    Configuration config = {{"metadata.broker.list", brokers}, {"acks", "1"}};
    producer_ = std::unique_ptr<BufferedProducer<std::string>>(
        new BufferedProducer<std::string>(config));
    msg_builder_ = std::unique_ptr<MessageBuilder>(new MessageBuilder(topic));
  }

  ~KafkaProducer() = default;

  void AddMessage(const std::string& message) {
    std::vector<std::string> split_msgs;
    split(message, '\n', split_msgs);
    for (const auto& s : split_msgs) {
      if (!s.empty()) {
        msg_builder_->payload(s);
        producer_->add_message(*msg_builder_);
      }
    }
    producer_->flush();
  }

  inline std::string topic() { return topic_; }

 private:
  std::string brokers_;
  std::string topic_;
  std::unique_ptr<MessageBuilder> msg_builder_;
  std::unique_ptr<BufferedProducer<std::string>> producer_;
};

class KafkaSink {
 public:
  typedef char char_type;
  typedef boost::iostreams::sink_tag category;

  explicit KafkaSink(std::shared_ptr<KafkaProducer> producer)
      : producer_(producer) {}

  std::streamsize write(const char_type* s, std::streamsize n) {
    std::string str(s, n);
    producer_->AddMessage(str);
    return n;
  }

 private:
  std::shared_ptr<KafkaProducer> producer_;
};

#endif  // EXAMPLES_GNN_SAMPLER_KAFKA_PRODUCER_H_
