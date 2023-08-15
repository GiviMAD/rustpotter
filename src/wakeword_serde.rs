use std::{fs::File, io::BufReader};

use ciborium::{de, ser};
use serde::{de::DeserializeOwned, Serialize};

pub trait SerializableWakeword: Serialize {
    fn save_to_file(&self, path: &str) -> Result<(), String> {
        let mut file = match File::create(path) {
            Ok(it) => it,
            Err(err) => {
                return Err("Unable to open file ".to_owned() + path + ": " + &err.to_string())
            }
        };
        ser::into_writer(self, &mut file).map_err(|err| err.to_string())?;
        Ok(())
    }
    fn save_to_buffer(&self) -> Result<Vec<u8>, String> {
        let mut bytes: Vec<u8> = Vec::new();
        ser::into_writer(self, &mut bytes).map_err(|err| err.to_string())?;
        Ok(bytes)
    }
}
pub trait DeserializableWakeword: DeserializeOwned + Sized {
    fn load_from_file(path: &str) -> Result<Self, String> {
        let file = match File::open(path) {
            Ok(it) => it,
            Err(err) => {
                return Err("Unable to open file ".to_owned() + path + ": " + &err.to_string())
            }
        };
        let reader = BufReader::new(file);
        Ok(de::from_reader(reader).map_err(|err| err.to_string())?)
    }
    fn load_from_buffer(buffer: &[u8]) -> Result<Self, String> {
        let reader = BufReader::new(buffer);
        Ok(de::from_reader(reader).map_err(|err| err.to_string())?)
    }
}
