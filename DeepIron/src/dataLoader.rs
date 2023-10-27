use std::path::Path;
use polars::prelude::*;


trait dataFrameTransformer {
    fn transform(
        &mut self,
        columns: &[&usize],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<()>;
}


impl DataFrameTransformer for DataFrame {
    fn transform(&mut self, columns: &[&usize], unary_function: impl Fn(&Series) -> Series) -> Result<()> {
        for (_, series) in self.get_columns() {
            let transformed_series = unary_function(series)?;
            self.with_column(transformed_series)?;
        }
        Ok(());
    }
}

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