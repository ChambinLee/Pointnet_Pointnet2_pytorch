## def main(args)

- ```python
  def log_string(str):
      logger.info(str)
      print(str)
  ```

  - 该函数用于记录日志，其原理暂不清楚。其是将传入字符串作为日志内容吗？

- ```python
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  ```

  - 该行用于用于当系统有多个可用的GPU时，将右边指定的GPU序号作为当前程序的可见GPU，并依次作为当前程序的0号、1号……可见GPU。详细说明见[博客](https://www.cnblogs.com/ying-chease/p/9473938.html)。

