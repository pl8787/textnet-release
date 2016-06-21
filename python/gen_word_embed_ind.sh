#! /bin/sh
echo "Generate unupdate word embeddings."
cat $1 | awk '{print $1" 0"}' > $1.ind
