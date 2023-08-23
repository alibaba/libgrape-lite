## z3 solver Getting started

Git clone z3 solver and run the following in the top level directory of the Z3 repository.

``
sudo git clone https://github.com/Z3Prover/z3.git

```
cd z3
sudo mkdir build
cd build
sudo cmake -G "Unix Makefiles" ../
sudo make -j4 # Replace 4 with an appropriate number
sudo make install

```
## libz3java.so
sudo cp libz3java.so /usr/lib

## ANTLR4 Getting started

linux
```
cd /usr/local/lib

wget https://www.antlr.org/download/antlr-4.9.2-complete.jar

#sudo vim ~/.bashrc

export CLASSPATH=".:/usr/local/lib/antlr-4.9.2-complete.jar:$CLASSPATH"

alias antlr4='java -jar /usr/local/lib/antlr-4.9.2-complete.jar'

alias grun='java org.antlr.v4.gui.TestRig'

#:wq!

#sudo source ~/.bashrc

### choose ingress engine by automanted

cd /examples/analytical_apps/antlr/src/transeng

sudo javac *.java -classpath ../../lib/antlr-4.9.2-complete.jar #编译transeng目录下的.java文件

sudo jar -cvf translation.jar translation.class  #将编译生成的translation.class压缩到translation.jar，jar包
#run translation
sudo java -cp .:../../lib/antlr-4.9.2-complete.jar translation   #运行translation进行翻译
-----------------------------------------------------
cd ../choose/egine

sudo javac *.java -classpath ../../lib/com.microsoft.z3.jar  #编译egine目录下的.java文件

sudo jar -cvf Usage.jar Usage.class   #将编译生成的Usage.class压缩到Usage.jar，jar包
## run ingress
sudo java -cp .:../../lib/com.microsoft.z3.jar Usage    ##运行Usage选择引擎，并且执行Ingress

