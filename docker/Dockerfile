FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV TZ="Asia/Shanghai"

EXPOSE 2222

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt update && apt install sudo openssh-server \
                      apt-transport-https ca-certificates gnupg software-properties-common wget curl \
                      openmpi-bin libopenmpi-dev libgoogle-glog-dev libgflags-dev libgtest-dev libssl-dev \
                      libboost-dev libboost-system-dev \
                      libboost-iostreams-dev libboost-serialization-dev \
                      libboost-filesystem-dev libboost-timer-dev libboost-chrono-dev \
                      iputils-ping net-tools vim git tmux gdb -y && \
                      update-alternatives --install /usr/bin/python python /usr/bin/python3 10

RUN mv /etc/ssh/sshd_config /etc/ssh/sshd_config.bak && \
    cat /etc/ssh/sshd_config.bak | sed -e 's/#Port 22/Port 2222/g' -e 's/#PasswordAuthentication yes/PasswordAuthentication no/g' > /etc/ssh/sshd_config && \
    echo "PubkeyAcceptedKeyTypes=+ssh-dss" >> /etc/ssh/sshd_config && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa

RUN rm -rf /etc/apt/sources.list.d/cuda.list && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && apt update && \
    apt install -y cmake

RUN wget http://gosspublic.alicdn.com/ossutil/1.7.6/ossutil64 && mv ossutil64 /usr/bin/ && chmod a+x /usr/bin/ossutil64

RUN groupadd -g 500 admin && useradd -rm -d /home/admin -s /bin/bash -g admin -G sudo -u 500 admin -p "$(openssl passwd -1 grape_dev)"
RUN echo "admin     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER admin
WORKDIR /home/admin

RUN ssh-keygen -t rsa -N ""  -f ~/.ssh/id_rsa && cp ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && echo "ssh-dss AAAAB3NzaC1kc3MAAACBAKfLOXKv1iPugbdydqeItxZPluN5j0jsuKY0rhSkOHAyK2AJimorAdIAYwKtx/CK6tEK18gjxKR44jtCtti1NcLTXmHyiV/XL8U087CFySGYcDk8hLbvZKEESpzXRSaxZA2n+fEEK/qbk+nnwZncmMWs8zCnf0rCKLDyNOYBd1vfAAAAFQDc+G7efWus1p6rr8EZ7ZYmXtLRcwAAAIBULgTrT+bnGv1csDxXtHXf0uArT41L8qh001Xu7xr0iUd77BW8pvZmu21+pxmWiM7aBHEOYD0Ju75foNNetnSsVocSvanZCPEzqCxjn7UaW7jzWYlJTi+Tj4UGNNprK6x1WS/cQS6fHAdF2QAo2zKpsK2kZPyBZF3BQe+oLQpEqAAAAIAF47fRDAaMUJqRqAthdLMVDALXztBi3esLwPIqhEQgYcuifsvEJDlyOHgYyem5CVHsXU9yTNRWwRLEEXbcodL4zaNlVHyu6Dra6p2TQK/SAsVpLHEFoUkewTcBmWl/M+QL/xDIH4A3fSXR5ZkWLqHJ6vcXGJDPQ4izrtoL0CBsBQ== mengke.mk@alibaba-inc.com" >> ~/.ssh/authorized_keys && echo "ssh-dss AAAAB3NzaC1kc3MAAACBANQPHgdZ2wyjQ+kpG9KbPuh7ZCkXwbd0Qar8/WuhPkE0Xyl72jgcqk9ER3rZ2iTzTLktwoCbmUcPhFntHq9bX3m30w5Pbx1xLUMeXebJiWv45gBIK7p/O8/3/MvdKPjrHPw+WNyMvabEB+vAPCazoc/nKZoVN3b78fj3FvVS0VYbAAAAFQD3on9sE2S/rvgxApVVhFZYehfQowAAAIEAoVvjJZwDuhXim7SXZZA3d2W1KEd36PC9cLGN0BnMc5xbTTZxIU2KCALzdGfqWaFFZNVJAv2zUI24eDjuAugyeXGiL1CcqqT/L1b1BhM7s2yo+6uzjN21IMvAvCi3xqVR1MYdc7Uu0TAVI5FFhgi40GnFG1EhCbBt+oBpY29pXwAAAACAPo94j2zXG1QARckHeTzsHo+fqslhk6nBStL3TzRH6G4nIc2j7GRWXx0f3l1wsfl9Q8Oig5w+KPwg4wBWCjFujzI0KG2YToU91umnTqLoMtS+fg3qlgkl+O1/PcVwkIFguzH7ICAU8SHJ2Kl8dGdjQqP2LXh9cwbUpXoVXL+FdUg= guanyi.gl@alibaba-inc.com" && echo "ssh-dss AAAAB3NzaC1kc3MAAACBAJi792xrSCHqfOueQrdR7/VlLSJQztgIJ8sKCfftvgXPbfEVPswWXHH3kVAM6/Z9gJJ4CKoAl2e6M9ylA0HqjTgjNEJWOVTAK8WBzEHTIdb+Dwldx79N/JE3/z0zIIz9FZ4jvowiaX21cSCpRfDGxbH/RYn3um2Ks4pNHPDpGKVZAAAAFQC2fcew5t8csOiVvhPtJaPPvSYQTQAAAIEAhob6GroELsijggDud7kw0o1zsJkMoDrJzGP7K35l5pcc2rClsC8tcR/6HjCmUID91V8xqVphGQEqUVa9yo31Eo6WKBWOlEa0RkV8pMIbg79tUSNkFeOQG5xm6DL2Iarl7RNIohxc3muXJ2dQUIfmBwc3oI56YbyossTV6F5lxCAAAACAOwaV/6lidlMAXheUNVoI1DhYn9D6rpOhd9MzzBjk/NtPZY8041QOguXuVxITVm8VoowaZmWfq2C96VQINeQyg/KBnaZlGSPwjLEeDDCw29i1VoQ50Y+1+UO78gpU+/baYWXH5zop6xSDZQewLNKCLLAKkRIiHvfAeWfc6s81Jok= tq273703@alibaba-inc.com" >> ~/.ssh/authorized_keys
