#include <opencv2/highgui/highgui.hpp>

using namespace cv;

//Original_LBP
Mat get_original_LBP_feature(Mat img){
  Mat result;
  result.create(img.rows - 2, img.cols -2, img.type());
  result.setTo(0);
  for (int i = 1; i < img.rows - 1; i++){
    for (int j = 1; j < img.cols -1; j++){
      uchar center = img.at<uchar>(i, j);
      uchar lbpcode = 0;
      lbpcode |= (img.at<uchar>(i - 1, j - 1) >= center) << 7;
      lbpcode |= (img.at<uchar>(i - 1, j) >= center) << 6;
      lbpcode |= (img.at<uchar>(i - 1, j + 1) >= center) << 5;
      lbpcode |= (img.at<uchar>(i, j -1) >= center) << 4;
      lbpcode |= (img.at<uchar>(i, j + 1) >= center) << 3;
      lbpcode |= (img.at<uchar>(i + 1, j - 1) >= center) << 2;
      lbpcode |= (img.at<uchar>(i + 1, j) >= center) << 1;
      lbpcode |= (img.at<uchar>(i + 1, j + 1) >= center) << 0;
      result.at<uchar>(i - 1, j - 1) = lbpcode;
    }
  }
  return result;
}

//Circular_LBP_feature
Mat get_circular_LBP_feature(Mat img, int radius, int neighbors)
{
  Mat result;
  result.create(img.rows - radius * 2, img.cols - radius * 2, img.type());
  result.setTo(0);
  //循环处理每个像素
  for(int i=radius;i<img.rows-radius;i++)
  {
      for(int j=radius;j<img.cols-radius;j++)
      {
          //获得中心像素点的灰度值
          uchar center = img.at<uchar>(i,j);
          uchar lbpCode = 0;
          for(int k=0;k<neighbors;k++)
          {
              //根据公式计算第k个采样点的坐标，这个地方可以优化，不必每次都进行计算radius*cos，radius*sin
              float x = i + static_cast<float>(radius * \
                  cos(2.0 * CV_PI * k / neighbors));
              float y = j - static_cast<float>(radius * \
                  sin(2.0 * CV_PI * k / neighbors));
                //根据取整结果进行双线性插值，得到第k个采样点的灰度值
                //1.分别对x，y进行上下取整
                int x1 = static_cast<int>(floor(x));
                int x2 = static_cast<int>(ceil(x));
                int y1 = static_cast<int>(floor(y));
                int y2 = static_cast<int>(ceil(y));

                //将坐标映射到0-1之间
                float tx = x - x1;
                float ty = y - y1;
                //根据0-1之间的x，y的权重计算公式计算权重
                float w1 = (1-tx) * (1-ty);
                float w2 =    tx  * (1-ty);
                float w3 = (1-tx) *    ty;
                float w4 =    tx  *    ty;
                //3.根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = img.at<uchar>(x1,y1) * w1 + img.at<uchar>(x1,y2) *w2 + img.at<uchar>(x2,y1) * w3 +img.at<uchar>(x2,y2) *w4;
                //通过比较获得LBP值，并按顺序排列起来
                lbpCode |= (neighbor>center) <<(neighbors-k-1);
            }
            result.at<uchar>(i-radius,j-radius) = lbpCode;
        }
    }
  return result;
}

//Rotation_Invariant_LBP_feature
Mat get_rotation_invariant_LBP_feature(Mat img, int radius, int neighbors)
{
  Mat result;
  result.create(img.rows - radius * 2, img.cols - radius * 2, img.type());
  result.setTo(0);
  for(int k=0;k<neighbors;k++)
    {
        //计算采样点对于中心点坐标的偏移量rx，ry
        float rx = static_cast<float>(radius * cos(2.0 * CV_PI * k / neighbors));
        float ry = -static_cast<float>(radius * sin(2.0 * CV_PI * k / neighbors));
        //为双线性插值做准备
        //对采样点偏移量分别进行上下取整
        int x1 = static_cast<int>(floor(rx));
        int x2 = static_cast<int>(ceil(rx));
        int y1 = static_cast<int>(floor(ry));
        int y2 = static_cast<int>(ceil(ry));
        //将坐标偏移量映射到0-1之间
        float tx = rx - x1;
        float ty = ry - y1;
        //根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        float w1 = (1-tx) * (1-ty);
        float w2 =    tx  * (1-ty);
        float w3 = (1-tx) *    ty;
        float w4 =    tx  *    ty;
        //循环处理每个像素
        for(int i=radius;i<img.rows-radius;i++)
        {
            for(int j=radius;j<img.cols-radius;j++)
            {
                //获得中心像素点的灰度值
                uchar center = img.at<uchar>(i,j);
                //根据双线性插值公式计算第k个采样点的灰度值
                float neighbor = img.at<uchar>(i+x1,j+y1) * w1 + img.at<uchar>(i+x1,j+y2) *w2 + img.at<uchar>(i+x2,j+y1) * w3 +img.at<uchar>(i+x2,j+y2) *w4;
                //LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                result.at<uchar>(i-radius,j-radius) |= (neighbor>center) <<(neighbors-k-1);
            }
        }
    }
    //进行旋转不变处理
    for(int i=0;i<result.rows;i++)
    {
        for(int j=0;j<result.cols;j++)
        {
            uchar currentValue = result.at<uchar>(i,j);
            uchar minValue = currentValue;
            for(int k=1;k<neighbors;k++)		//循环左移
            {
                uchar temp = (currentValue>>(neighbors-k)) | (currentValue<<k);
                if(temp < minValue)
                {
                    minValue = temp;
                }
            }
            result.at<uchar>(i,j) = minValue;
        }
    }
    return result;
}

int main(int argc, char* argv[])
{
  Mat src = imread(argv[1], 0);
  Mat dst = get_original_LBP_feature(src);
  Mat odst1 = get_circular_LBP_feature(src, 1, 8);
  //Mat odst4 = get_circular_LBP_feature(src, 4, 8);
  Mat rif8 = get_rotation_invariant_LBP_feature(src, 1, 8);
  Mat rif6 = get_rotation_invariant_LBP_feature(src, 1, 6);


  imshow("原始图片", src);
  imshow("原始LBP", dst);
  imshow("圆形LBP", odst1);
  //imshow("圆形LBP4", odst4);
  imshow("旋转不变LBP", rif8);
  //imshow("旋转不变LBP6", rif6);

  waitKey(0);
  return 0;
}
