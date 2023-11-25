pub mod dataLoader;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_load_csv() {
        let path = Path::new("test_data.csv");
        let result = dataLoader::DataLoader::loadCSV(path);

        // Check if the result is Ok and the DataFrame is not empty
        assert!(result.is_ok());
        let df = result.unwrap();
        assert!(!df.is_empty());
    }

    #[test]
    fn test_transform_by_col() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1, 2, 3])]);

        // Apply a transformation
        df.transformByCol(&["col1"], |s| s * 2).unwrap();

        // Check if the transformation is applied correctly
        let expected_result = DataFrame::new(vec![Series::new("col1", &[2, 4, 6])]);
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_split() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1, 2, 3, 4, 5])]);

        // Split the DataFrame
        let (train, test) = df.split(0.8).unwrap();

        // Check if the split is done correctly
        assert_eq!(train.height(), 4);
        assert_eq!(test.height(), 1);
    }

    #[test]
    fn test_z_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1, 2, 3, 4, 5])]);

        // Z-normalize the column
        df.zNormCols(&["col1"]).unwrap();

        // Check if the z-normalization is done correctly
        let expected_result = DataFrame::new(vec![Series::new("col1", &[-1.41421, -0.70711, 0.0, 0.70711, 1.41421])]);
        assert_eq!(df, expected_result);
    }

    #[test]
    fn test_min_max_norm_cols() {
        // Create a simple DataFrame for testing
        let mut df = DataFrame::new(vec![Series::new("col1", &[1, 2, 3, 4, 5])]);

        // Min-max normalize the column
        df.minMaxNormCols(&["col1"]).unwrap();

        // Check if the min-max normalization is done correctly
        let expected_result = DataFrame::new(vec![Series::new("col1", &[0.0, 0.25, 0.5, 0.75, 1.0])]);
        assert_eq!(df, expected_result);
    }
}
