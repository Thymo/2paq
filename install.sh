###### Installs project dependencies
poetry install

###### Downloads GENRE models and data ######

cd data
mkdir genre

### KILT prefix tree
curl -O http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl

### GENRE ED model
curl -O http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz
tar -zxvf hf_entity_disambiguation_blink.tar.gz

### GENRE DR model
curl -O http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz
tar -zxvf hf_wikipage_retrieval.tar.gz