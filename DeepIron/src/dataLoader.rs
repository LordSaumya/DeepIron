use polars::prelude::*;
use std::path::Path;
/// Trait for DataFrame transformation, including transforms and splits.
trait DataFrameTransformer {
    fn transformByCol(
        &mut self,
        columns: &[&str],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<(), PolarsError>;

    fn split(&mut self, train_size: f64) -> Result<(DataFrame, DataFrame), PolarsError>;
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
    /// * `Result<(), PolarsError>` - An empty Result.
    ///
    /// # Example
    ///
    /// ```
    /// df.transformByCol(&[0, 1], |series| series * 2);
    /// ```
    fn transformByCol(
        &mut self,
        columns: &[&str],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<(), PolarsError> {
        for col in columns {
            // get the column and transform it
            let series = self.column(col)?;
            let transformed_series = unary_function(series);
            self.with_column(transformed_series)?;
        }
        Ok(())
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
    fn split(&mut self, train_size: f64) -> Result<(DataFrame, DataFrame), PolarsError> {
        // get the number of rows in the DataFrame
        let num_rows = self.height();

        // get the number of rows in the training DataFrame
        let train_num_rows = (num_rows as f64 * train_size) as i64;

        // get the training DataFrame
        let train = self.slice(0, train_num_rows as usize);
        
        // get the testing DataFrame
        let test = self.slice(train_num_rows, num_rows);
        Ok((train, test))
    }
}

/// A set of functions for loading and transforming data into a Polars DataFrame.
pub mod DataLoader {
    use polars::prelude::{PolarsArray, PolarsError};

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
    pub fn loadCSV(path: &Path) -> Result<DataFrame, PolarsError> {
        let path = Path::new(path);
        let df = match DataFrame::read_csv(path) {
            Ok(df) => df,
            Err(e) => PolarsError::from(e)
        };
        Ok(df)
    }
}

/// A set of functions that return commonly-used series -> series functions for data transformations.
///
/// 
/// # Example
///
/// ```
/// df.transform(&[0, 1], TransformerFunctions::identity());
/// df.transform(&[0, 1], TransformerFunctions::power(2));
/// ```
pub mod TransformerFunctions {
    /// Return a function that returns the identity of a Series.
    fn identity() -> impl Fn(&Series) -> Series {
        move |series| series.clone()
    }

    /// Return a function that returns the power of a Series.
    /// 
    /// # Arguments
    /// 
    /// * `power` - The power to raise the Series to.
    /// 
    /// # Returns
    /// 
    /// * `impl Fn(&Series) -> Series` - A function that takes a Series and returns a Series.
    fn power(power: f64) -> impl Fn(&Series) -> Series {
        move |series| series.pow(power)
    }

    /// Return a function that returns the log of a Series.
    /// 
    /// # Arguments
    /// 
    /// * `base` - The base of the log.
    /// 
    /// # Returns
    /// 
    /// * `impl Fn(&Series) -> Series` - A function that takes a Series and returns a Series.
    fn log(base: f64) -> impl Fn(&Series) -> Series {
        move |series| series.log(base)
    }
}