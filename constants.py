FUSION_APP_NAME = 'bestbuy'

# Product catalog Solr collection
COLLECTION_NAME = 'bestbuy'

# Name of Fusion query pipeline (needs to output [features] as field, see setup instructions)
FUSION_QUERY_PIPELINE_NAME = 'bestbuy'

# Signals Solr collection
SIGNALS_COLLECTION_NAME = 'bestbuy_signals'

# Base fq applied to very signals query
SIGNALS_BASE_FQ = 'type:click'

# Name of Solr LTR feature store
FEATURE_STORE_NAME = "my_efi_feature_store"

# Output path of feature extraction
FEATURE_VECTORS_PATH = '/tmp/feature_vectors'

# Number of (most recent) clickthrough signals to run feature extraction on
# Chosen more-or-less arbitrarily. Feature extraction and model training is expensive
# so we want to choose a statistically significant number and no more than that.
SIGNALS_FEATURE_EXTRACTION_LIMIT = 5000

# I have chosen STAGE_A_TOP_N more ore less arbitrarily. An optimal STAGE_A_TOP_N depends on the particular balance between
# recall and performance for a given application.
STAGE_A_TOP_N = 100