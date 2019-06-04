Ever wanted to parse your developer logbook but wanted to separate code from your other remarks?

Bayesian classifier. Pretrained on Python code and SQL.


docker run --rm -v $(pwd):/code -it nltk bash
docker run --rm -v $(pwd)/pickled_models:/code/pickled_models -v $(pwd)/landing:/landing -v $(pwd)/processed:/processed nltk

prototype TODO:
create docker environment with nltk installed ✓
paste some python code in ./code ✓
paste some HQL in ./code
use some nltk demo dataset for ./notcode ✓
write training code ✓
pickle the model ✓
write classifier ✓
make it interface with a landing zone and output zone
