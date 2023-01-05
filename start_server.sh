# agree to EULA
sed -i 's/false/true/' eula.txt
update-alternatives --set java /usr/lib/jvm/java-17-openjdk-amd64/bin/java
java -Xmx1024M -Xms1024M -jar server.jar nogui