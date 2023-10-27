use std::path::Path;
use polars::prelude::*;


/// Trait for DataFrame transformation, including transforms and splits.
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

/// Implement DataFrameTransformer for DataFrame.
impl DataFrameTransformer for DataFrame {
    /// Transform the DataFrame by column.
    /// 
    /// # Arguments
    /// 
    /// * `columns` - A slice of column indices to transform.
    /// * `unary_function` - A function that takes a Series and returns a Series.
    /// 
    /// # Returns
    /// 
    /// * `Result<()>` - An empty Result if successful.
    /// 
    /// # Example
    /// 
    /// ```
    /// df.transformByCol(&[0, 1], |series| series * 2);
    /// ```
    fn transformByCol(&mut self, columns: &[&usize], unary_function: impl Fn(&Series) -> Series) -> Result<()> {
        for col in columns {
            // check if column index is out of range
            if *col >= self.width() {
                return Err(PolarsError::OutOfBounds("Column index out of range".into()).into());
            }

            // get the column and transform it
            let series = self.column(*col)?;
            let transformed_series = unary_function(series)?;
            self.with_column(transformed_series)?;
        }
        Ok(());
    }

    /// Split the DataFrame into two DataFrames.
    /// 
    /// # Arguments
    /// 
    /// * `train_size` - The size of the training DataFrame as a percentage of the original DataFrame.
    /// 
    /// # Returns
    /// 
    /// * `Result<(DataFrame, DataFrame)>` - A tuple of DataFrames, the first being the training DataFrame and the second being the testing DataFrame.
    /// 
    /// # Example
    /// 
    /// ```
    /// let (train, test) = df.split(0.8);
    /// ```
    fn split(&mut self, train_size: f64) -> Result<(DataFrame, DataFrame)> {
        let (train, test) = self.shuffle(train_size)?;
        Ok((train, test))
    }
}

/// A set of functions for loading and transforming data into a Polars DataFrame.
pub mod dataLoader {
    /// Load a CSV file into a DataFrame.
    /// 
    /// # Arguments
    /// 
    /// * `path` - The path to the CSV file.
    /// 
    /// # Returns
    /// 
    /// * `Result<DataFrame>` - A DataFrame containing the data from the CSV file.
    /// 
    /// # Example
    /// 
    /// ```
    /// let df = loadCSV("data.csv");
    /// ```
    pub fn loadCSV(path: &Path) -> Result<DataFrame> {
        let path = Path::new(path);
        let df = match DataFrame::read_csv(path) {
            Ok(df) => df,
            Err(e) => return Err(e.into()),
        };
        Ok(df);
    }
}