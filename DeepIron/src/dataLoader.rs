use std::path::Path;
use polars::prelude::*;


trait DataFrameTransformer {
    fn transformByCol(
        &mut self,
        columns: &[&usize],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<()>;

    fn split(
        &mut self,
        train_size: f64
    ) -> Result<(DataFrame, DataFrame)>;
}

impl DataFrameTransformer for DataFrame {
    fn transformByCol(&mut self, columns: &[&usize], unary_function: impl Fn(&Series) -> Series) -> Result<()> {
        for (_, series) in self.get_columns() {
            let transformed_series = unary_function(series)?;
            self.with_column(transformed_series)?;
        }
        Ok(());
    }

    fn split(&mut self, train_size: f64) -> Result<(DataFrame, DataFrame)> {
        let (train, test) = self.shuffle(train_size)?;
        Ok((train, test))
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