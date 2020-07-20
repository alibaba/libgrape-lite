/*
 * Copyright 2015 Delft University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#ifdef GRANULA

namespace granula {
    using namespace std;

    class operation {
        public:
            string operationUuid;
            string actor_type;
            string actor_id;
            string mission_type;
            string mission_id;

            operation(string a_type, string a_id, string m_type, string m_id) {
                operationUuid = generateUuid();
                actor_type = a_type;
                actor_id = a_id;
                mission_type = m_type;
                mission_id = m_id;
            }

            string generateUuid() {
                long uuid;
                if (sizeof(int) < sizeof(long))
                uuid = (static_cast<long>(rand()) << (sizeof(int) * 8)) | rand();
                return to_string(uuid);
            }

            string getOperationInfo(string infoName, string infoValue) {
                return "GRANULA - OperationUuid:" + operationUuid + " " +
                   "ActorType:" + actor_type + " " +
                   "ActorId:" + actor_id + " " +
                   "MissionType:" + mission_type + " " +
                   "MissionId:" + mission_id + " " +
                   "InfoName:" + infoName + " " +
                   "InfoValue:" + infoValue + " " +
                   "Timestamp:" + getEpoch();
            }

            string getEpoch() {
                return to_string(chrono::duration_cast<chrono::milliseconds>
                    (chrono::system_clock::now().time_since_epoch()).count());
            }
    };


    void error(const char *msg) {
        perror(msg);
        exit(0);
    }

    void sendMonitorMessage(std::string message) {
        fprintf(stdout, "Communicating with Granula monitor.\n");
        // server: for lookup of server address
        // serv_addr: serv_addr
        // portno: port number
        // buffer: message content
        // n = message length counter
        // sockfd: socket tool
        int sockfd, portno, n;
        struct sockaddr_in serv_addr;
        struct hostent *server;
        char buffer[512];

        portno = 2656;

        // setup socket object
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            fprintf(stdout, "Connection Failed: cannot open socket.\n");
            return;
        }

        // set up server by hostname
        server = gethostbyname("localhost");
        if (server == NULL) {
            fprintf(stdout, "Connection Failed: no such host.\n");
            close(sockfd);
            return;
        }

        // set up server address using server object
        bzero((char *) &serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        bcopy((char *) server->h_addr, (char *) &serv_addr.sin_addr.s_addr, server->h_length);

        serv_addr.sin_port = htons(portno);

        // connect socket to server address
        if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
            fprintf(stdout, "Connection Failed: cannot connect to server.\n");
            close(sockfd);
            return;
        }

        // buffer message before send
        bzero(buffer, 512);
        message.copy(buffer, message.length());
        std::cout.write(buffer, message.length());
        std::cout.put('\n');

        // write to socket
        n = write(sockfd, buffer, strlen(buffer));
        if (n < 0) {
             fprintf(stdout, "Connection Failed: cannot write to socket.\n");
             close(sockfd);
        }

        close(sockfd);
    }


    void linkProcess(int processId, std::string jobId) {
        std::string message = "{\"type\":\"Monitor\", \"state\":\"LinkProcess\", \"jobId\":\""+jobId+"\", \"processId\":\""+std::to_string(processId)+"\"}";
        sendMonitorMessage(message);
    }

    void linkNode(std::string jobId) {
        std::string message = "{\"type\":\"Monitor\", \"state\":\"LinkNode\", \"jobId\":\""+jobId+"\"}";
        sendMonitorMessage(message);
    }


    void startMonitorProcess(int processId) {
        std::string message = "{\"type\":\"Monitor\", \"state\":\"StartMonitorProcess\", \"processId\":\""+std::to_string(processId)+"\"}";
        sendMonitorMessage(message);
    }

    void stopMonitorProcess(int processId) {
        std::string message = "{\"type\":\"Monitor\", \"state\":\"StopMonitorProcess\", \"processId\":\""+std::to_string(processId)+"\"}";
        sendMonitorMessage(message);
    }

}

#endif