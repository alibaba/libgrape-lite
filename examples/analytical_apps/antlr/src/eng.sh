cd ./transeng
sudo javac *.java -classpath ../../lib/antlr-4.9.2-complete.jar
sudo jar -cvf translation.jar translation.class
# 用户需要指定 "../../../pagerank/pagerank_ingress.h"为对应的图算法路径
sudo java -cp .:../../lib/antlr-4.9.2-complete.jar translation ../../../pagerank/pagerank_ingress.h
cd ../choose_egine/
sudo javac *.java -classpath ../../lib/com.microsoft.z3.jar
sudo jar -cvf Usage.jar Usage.class
# 用户需要指定 "pagerank"为自己的图算法名称
sudo java -cp .:../../lib/com.microsoft.z3.jar Usage -application pagerank -vfile /home/yongze/dataset/test_o.txt -efile /home/yongze/dataset/test.txt -directed false -cilk true -termcheck_threshold 0.000001 -app_concurrency 1 -sssp_source 1  -out_prefix /home/yongze/dataset/output
#php，pagerank通过测试
#pagerank,pagerank  通过测试 