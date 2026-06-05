import kagglehub

# Download latest version
path = kagglehub.dataset_download("gzipchrist/leetcode-problem-dataset")

print("Path to dataset files:", path)