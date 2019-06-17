module.exports = {
  entry: './example/usage.js',
  output: {
    filename: 'bundle.js'
  },
  devServer: {
    contentBase: "./example"
  },
  resolve: {
    extensions: ['.webpack.js', '.ts', '.js']
  },
  module: {
    loaders: [
      { test: /\.tsx?$/, loader: 'ts-loader' }
    ]
  }
}
