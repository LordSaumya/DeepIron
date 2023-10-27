use std::path::Path;
use polars::prelude::*;

pub mod dataLoader {
    pub fn loadCSV(path: &Path) -> Result<DataFrame> {
        let path = Path::new(path);
        let df = match DataFrame::read_csv(path) {
            Ok(df) => df,
            Err(e) => return Err(e.into()),
        };
        Ok(df);
    }
}