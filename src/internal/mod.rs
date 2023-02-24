pub(self) const MIN_I16_ABS_VAL: f32 = (i16::MIN as f32) * -1.;
pub(self) const MIN_I16_VAL: f32 = i16::MIN as f32;
pub(self) const MAX_I16_VAL: f32 = i16::MAX as f32;

mod band_pass_filter;
mod comparator;
mod dtw;
mod encoder;
mod feature_extractor;
mod feature_normalizer;
mod gain_normalizer_filter;
pub(crate) use band_pass_filter::BandPassFilter;
pub(crate) use comparator::FeatureComparator;
pub(crate) use dtw::Dtw;
pub(crate) use encoder::WAVEncoder;
pub(crate) use feature_extractor::FeatureExtractor;
pub(crate) use feature_normalizer::FeatureNormalizer;
pub(crate) use gain_normalizer_filter::GainNormalizerFilter;
