# #!/bin/bash
set -e

echo "Downloading IATE archive..."
mkdir -p data/
cd data/
curl http://iate.europa.eu/downloadTbx.do -o IATE_download.zip
unzip -oq IATE_download.zip
mv IATE_export*.tbx IATE_export.tbx
rm *.zip
cd ../

echo "Downloading Stanford CoreNLP parser..."
mkdir -p lib/
cd lib/
VERSION=2018-02-27
curl -O https://nlp.stanford.edu/software/stanford-parser-full-$VERSION.zip -#
unzip -oq stanford-parser-full-$VERSION.zip
mv stanford-parser-full-$VERSION/ stanford-parser/
curl -O https://nlp.stanford.edu/software/stanford-postagger-full-$VERSION.zip -#
unzip -oq stanford-postagger-full-$VERSION.zip
mv stanford-postagger-full-$VERSION/ stanford-postagger/
rm *.zip
cd ../

echo "Downloading fastText [en] word vectors..."
mkdir -p fasttext/
cd fasttext/
curl -O https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip -#
unzip -oq crawl-300d-2M.vec.zip
rm crawl-300d-2M.vec.zip
mv crawl-300d-2M.vec cc.en.300.vec
for lang in bg cs da de el es et fi fr ga hr hu it la lt lv mt nl pl pt ro sk sl sv; do
  echo "Downloading fastText [$lang] word vectors..."
  curl -O https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.$lang.300.vec.gz -#
  gunzip -q cc.$lang.300.vec.gz
done
