use rustpotter::{
    ModelType, WakewordLoad, WakewordModel, WakewordModelTrain, WakewordModelTrainOptions,
    WakewordRef, WakewordRefBuildFromFiles, WakewordSave,
};

#[test]
fn it_creates_a_new_wakeword_from_samples() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let samples = vec![
        dir.to_owned() + "/tests/resources/oye_casa_g_1.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_2.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_3.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_4.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_5.wav",
    ];
    let n_samples = samples.len();
    let wakeword =
        WakewordRef::new_from_sample_files("oye casa".to_string(), None, None, samples, 5).unwrap();
    assert_eq!(
        wakeword.samples_features.len(),
        n_samples,
        "Sample features are extracted"
    );
}

#[test]
fn it_creates_a_new_wakeword_from_int_samples_which_saves_to_file() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let samples = vec![
        dir.to_owned() + "/tests/resources/oye_casa_g_1.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_2.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_3.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_4.wav",
        dir.to_owned() + "/tests/resources/oye_casa_g_5.wav",
    ];
    let wakeword =
        WakewordRef::new_from_sample_files("oye casa".to_string(), None, None, samples, 5).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    wakeword.save_to_file(&model_path).unwrap();
}

#[test]
fn it_creates_another_wakeword_from_int_samples_which_saves_to_file() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let samples = vec![
        dir.to_owned() + "/tests/resources/alexa.wav",
        dir.to_owned() + "/tests/resources/alexa2.wav",
        dir.to_owned() + "/tests/resources/alexa3.wav",
    ];
    let wakeword =
        WakewordRef::new_from_sample_files("alexa".to_string(), None, None, samples, 5).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/alexa.rpw";
    wakeword.save_to_file(&model_path).unwrap();
}

#[test]
fn it_creates_wakeword_from_float_samples_which_saves_to_file() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let samples = vec![
        dir.to_owned() + "/tests/resources/oye_casa_real_1.wav",
        dir.to_owned() + "/tests/resources/oye_casa_real_2.wav",
        dir.to_owned() + "/tests/resources/oye_casa_real_3.wav",
        dir.to_owned() + "/tests/resources/oye_casa_real_4.wav",
        dir.to_owned() + "/tests/resources/oye_casa_real_5.wav",
        dir.to_owned() + "/tests/resources/oye_casa_real_6.wav",
    ];
    let wakeword =
        WakewordRef::new_from_sample_files("oye casa".to_string(), None, None, samples, 5).unwrap();
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_real.rpw";
    wakeword.save_to_file(&model_path).unwrap();
}

#[test]
fn it_loads_a_wakeword_from_file() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let n_samples = 5;
    let model_path = dir.to_owned() + "/tests/resources/oye_casa_g.rpw";
    let wakeword = WakewordRef::load_from_file(&model_path).unwrap();
    assert_eq!(
        wakeword.samples_features.len(),
        n_samples,
        "Samples features number is correct"
    );
}

#[test]
fn it_train_a_new_wakeword_from_samples() {
    let dir = env!("CARGO_MANIFEST_DIR");
    let train_dir = dir.to_owned() + "/tests/resources/train";
    let test_dir = dir.to_owned() + "/tests/resources/test";
    let mfcc_size = 16;
    let train_config = WakewordModelTrainOptions::new(ModelType::Medium, 0.027, 10, 10, mfcc_size);
    let ww_model = WakewordModel::train_from_dirs(train_config, train_dir, test_dir, None).unwrap();
    assert_eq!(ww_model.labels.len(), 2, "Sample features are extracted");
    assert_eq!(ww_model.weights.len(), 6, "Model weights are created");
    assert_eq!(ww_model.train_size, 168, "Correct number of mfcc vectors");
    assert_eq!(ww_model.mfcc_size, mfcc_size, "Correct mfcc vector length");
}
