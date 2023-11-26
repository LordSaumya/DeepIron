pub mod dataLoader;

#[cfg(test)]
mod tests {
    use dataLoader::*;
    use polars::prelude::*;
    use std::path::Path;    
    use super::*;
    #[test]
    fn test_load_csv() {
        // Load the CSV file
        let path: &Path = Path::new("test/loadTest.csv");
        let result: Result<DataFrame, PolarsError> = dataLoader::DataLoader::loadCSV(path);

        println!("{:?}", result);

        // Check if the result is Ok and the DataFrame is not empty
        assert!(result.is_ok());
        let df: DataFrame = result.expect("Failed to load CSV file");
        assert!(!df.is_empty());
    }

    #[test]
    fn test_transform_by_col_custom_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df.transformByCol(&["col1"], |s: &Series| s * 2).unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame = DataFrame::new(vec![Series::new("col1", &[2.0, 4.0, 6.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_identity_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df.transformByCol(&["col1"], TransformerFunctions::identity()).unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_power_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0])]).unwrap();

        // Apply a transformation
        df = df.transformByCol(&["col1"], TransformerFunctions::power(2.0)).unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame = DataFrame::new(vec![Series::new("col1", &[1.0, 4.0, 9.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_transform_by_col_log_funct() {
        // Create a simple DataFrame for testing
        let mut df: DataFrame = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 4.0])]).unwrap();

        // Apply a transformation
        df = df.transformByCol(&["col1"], TransformerFunctions::log(2.0)).unwrap();

        // Check if the transformation is applied correctly
        let expected_result: DataFrame = DataFrame::new(vec![Series::new("col1", &[0.0, 1.0, 2.0])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_split() {
        // Create a simple DataFrame for testing
        let df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Split the DataFrame
        let (train, test) = df.split(0.8).unwrap();

        // Check if the split is done correctly
        assert_eq!(train.height(), 4);
        assert_eq!(test.height(), 1);
    }

    #[test]
    fn test_z_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Z-normalise the column
        df = df.zNormCols(&["col1"]).unwrap();

        let std: f64 = f64::sqrt(2.0);
        let mean: f64 = 3.0;

        // Check if the z-normalisation is done correctly
        let expected_result = DataFrame::new(vec![Series::new("col1", &[(1.0 - mean) / std, (2.0 - mean) / std, (3.0 - mean) / std, (4.0 - mean) / std, (5.0 - mean) / std])]).unwrap(); 
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_chained_transformations() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 4.0])]).unwrap();

        // Apply a transformation
        df = df
        .transformByCol(&["col1"], TransformerFunctions::power(2.0))
        .transformByCol(&["col1"], TransformerFunctions::log(2.0))
        .transformByCol(&["col1"], TransformerFunctions::identity())
        .zNormCols(&["col1"]).unwrap();
        
        let std: f64 = f64::sqrt(8.0/3.0);
        let mean: f64 = 2.0;    

        // Check if the transformation is applied correctly
        let expected_result = DataFrame::new(vec![Series::new("col1", &[(0.0 - mean) / std, (2.0 - mean) / std, (4.0 - mean) / std])]).unwrap();
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_min_max_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1.0, 2.0, 3.0, 4.0, 5.0])]).unwrap();

        // Min-max normalise the column
        df = df.minMaxNormCols(&["col1"]).unwrap();

        // Check if the min-max normalisation is done correctly
        let expected_result = DataFrame::new(vec![Series::new("col1", &[0.0, 0.25, 0.5, 0.75, 1.0])]).unwrap();
        assert_eq!(df, expected_result);
    }
}
