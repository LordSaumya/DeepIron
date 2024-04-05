use polars::frame::DataFrame;
use polars::prelude::*;
use polars::series::Series;

/// Trait for DataFrame transformation, including transforms and splits.
pub trait DataFrameTransformer {
    fn transform_cols(
        &self,
        columns: &[&str],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<DataFrame, PolarsError>;

    fn split(&self, train_size: f64) -> Result<(DataFrame, DataFrame), PolarsError>;

    fn z_norm_cols(&self, columns: &[&str]) -> Result<DataFrame, PolarsError>;

    fn min_max_norm_cols(&self, columns: &[&str]) -> Result<DataFrame, PolarsError>;

    fn get_col_by_index(&self, index: usize) -> Result<Series, PolarsError>;
}

/// Implement DataFrameTransformer for Result<DataFrame, PolarsError> for easier chaining of DataFrame transformations.
impl DataFrameTransformer for Result<DataFrame, PolarsError> {
    fn transform_cols(
        &self,
        columns: &[&str],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<DataFrame, PolarsError> {
        let df: &DataFrame = self.as_ref().unwrap();
        df.transform_cols(columns, unary_function)
    }

    fn split(&self, train_size: f64) -> Result<(DataFrame, DataFrame), PolarsError> {
        let df: &DataFrame = self.as_ref().unwrap();
        df.split(train_size)
    }

    fn z_norm_cols(&self, columns: &[&str]) -> Result<DataFrame, PolarsError> {
        let df: &DataFrame = self.as_ref().unwrap();
        df.z_norm_cols(columns)
    }

    fn min_max_norm_cols(&self, columns: &[&str]) -> Result<DataFrame, PolarsError> {
        let df: &DataFrame = self.as_ref().unwrap();
        df.min_max_norm_cols(columns)
    }

    fn get_col_by_index(&self, index: usize) -> Result<Series, PolarsError> {
        let df: &DataFrame = self.as_ref().unwrap();
        df.get_col_by_index(index)
    }
}

/// Implement DataFrameTransformer for DataFrame.
impl DataFrameTransformer for DataFrame {
    /// Transform the DataFrame by column.
    ///
    /// # Arguments
    ///
    /// * `columns` - A slice of columns to transform.
    /// * `unary_function` - A function that takes a Series and returns a Series.
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame, PolarsError>` - A DataFrame containing the transformed columns.
    ///
    /// # Example
    ///
    /// ```
    /// df = df.transform_cols(&["col1", "col2"], transformer_functions::identity());
    /// ```
    fn transform_cols(
        &self,
        columns: &[&str],
        unary_function: impl Fn(&Series) -> Series,
    ) -> Result<DataFrame, PolarsError> {
        let mut df: DataFrame = self.clone();
        for col in columns {
            // get the column and transform it
            let series: &Series = self.column(col)?;
            let transformed_series: Series = unary_function(series);
            df.with_column(transformed_series)?;
        }
        Ok(df)
    }

    /// Split the DataFrame into training and testing DataFrames.
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
    fn split(&self, train_size: f64) -> Result<(DataFrame, DataFrame), PolarsError> {
        let num_rows: usize = self.height();
        let train_num_rows: i64 = (num_rows as f64 * train_size) as i64;
        let train: DataFrame = self.slice(0, train_num_rows as usize);
        let test: DataFrame = self.slice(train_num_rows, num_rows);
        Ok((train, test))
    }

    /// Z-normalise the columns of the DataFrame.
    ///
    /// # Arguments
    ///
    /// * `columns` - A slice of column indices to z-normalise.
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame, PolarsError>` - A DataFrame containing the z-normalised columns.
    ///
    /// # Example
    ///
    /// ```
    /// df.z_norm_cols(&["col1", "col2"]);
    /// ```
    fn z_norm_cols(&self, columns: &[&str]) -> Result<DataFrame, PolarsError> {
        let mut df: DataFrame = self.clone();

        for col in columns {
            let series: &Series = self.column(col)?;
            let mean: f64 = series.mean().unwrap();
            let std: f64 = if let AnyValue::Float64(value) = series.std_as_series(0).get(0).unwrap()
            {
                value
            } else {
                panic!("Standard deviation is not F64");
            };
            let transformed_series: Series = (series - mean) / std;
            df.with_column(transformed_series)?;
        }
        Ok(df)
    }

    /// Min-max normalise the columns of the DataFrame.
    ///
    /// # Arguments
    ///
    /// * `columns` - A slice of columns to min-max normalise.
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame, PolarsError>` - A DataFrame containing the min-max normalised columns.
    ///
    /// # Example
    ///
    /// ```
    /// df.min_max_norm_cols(&["col1", "col2"]);
    /// ```
    fn min_max_norm_cols(&self, columns: &[&str]) -> Result<DataFrame, PolarsError> {
        let mut df: DataFrame = self.clone();

        for col in columns {
            let series: &Series = self.column(col)?;
            let min: f64 = series.min().unwrap();
            let max: f64 = series.max().unwrap();
            let transformed_series: Series = (series - min) / (max - min);
            df.with_column(transformed_series)?;
        }
        Ok(df)
    }

    /// Get a column of the DataFrame by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the column to get.
    ///
    /// # Returns
    ///
    /// * `Result<Series, PolarsError>` - The column as a Series.
    ///
    /// # Example
    ///
    /// ```
    /// let col = df.get_col_by_index(0);
    /// ```
    fn get_col_by_index(&self, index: usize) -> Result<Series, PolarsError> {
        let series: &[Series] = self.get_columns();
        Ok(series[index].clone())
    }
}

/// A set of functions for loading and transforming data into a Polars DataFrame.
pub mod data_loader_util {
    use polars::frame::DataFrame;
    use polars::prelude::{CsvReader, PolarsError, SerReader};
    use std::path::Path;

    /// Load a CSV file into a DataFrame.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the CSV file.
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame, PolarsError>` - A DataFrame containing the data from the CSV file.
    ///
    /// # Example
    ///
    /// ```
    /// let df = load_csv("data.csv");
    /// ```
    pub fn load_csv(path: &Path) -> Result<DataFrame, PolarsError> {
        let path: &Path = Path::new(path);
        let df: DataFrame = CsvReader::from_path(path)?
            .has_header(true)
            .finish()
            .unwrap();
        Ok(df)
    }
}

/// A set of functions that return commonly-used series -> series functions for data transformations.
///
///
/// # Example
///
/// ```
/// df.transform(&["col1", "col2"], TransformerFunctions::identity());
/// df.transform(&["col1", "col2"], TransformerFunctions::power(2));
/// ```
pub mod transformer_functions {
    use polars::prelude::*;
    use polars::series::Series;
    /// Return a function that returns the identity of a Series.
    ///
    /// # Returns
    ///
    /// * `impl Fn(&Series) -> Series` - A function that takes a Series and returns a Series.
    ///
    /// # Example
    ///
    /// ```
    /// df.transform(&["col1", "col2"], TransformerFunctions::identity());
    /// ```
    pub fn identity() -> impl Fn(&Series) -> Series {
        move |series: &Series| series.clone()
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
    pub fn power(power: f64) -> impl Fn(&Series) -> Series {
        return move |series: &Series| {
            let s_power: Series = series
                .f64()
                .expect("series was not an f64 dtype")
                .apply(|value| value.map(|value| value.powf(power)))
                .into();
            s_power
        };
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
    pub fn log(base: f64) -> impl Fn(&Series) -> Series {
        return move |series: &Series| {
            let s_log: Series = series
                .f64()
                .expect("series was not an f64 dtype")
                .apply(|value| value.map(|value| value.log(base)))
                .into();
            s_log
        };
    }
}
