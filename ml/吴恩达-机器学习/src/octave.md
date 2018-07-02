# ocatve使用

## 画图
+ `t = [0:0.01:0.98]` 取 0 到 0.98 的点 间隔为0.01
+  `y1 = sin(2*pi*4*t) plot(t,y)` 将t和y在图中画出
+ `hold on ` 保持当前画图
+ `plot(t,y2,'r')` r : 红色
+ `xlabel('time')` x轴标注
+ `ylabel('value')` y轴标注
+ `legend('sin','cos')` 图形标注 第一个sin,第二个cos
+ `title(my plot)` 给图添加标题
+ `print -dpng 'myPlot.png'`将图片保持到当前目录下的myPlot.png文件
+ `close` 关闭当前的画图
+ `figure(1)` 打开画图1
+ `subplot(1,2,1);`  将画图分成 1 * 2个网格, 使用第一个,后面使用plot方法画在第一个上
+ `subplot(1,2,2);`  将画图分成 1 * 2个网格, 使用第二个,后面使用plot方法画在第二个上
+ `axis([0.5 1 -1 1])` 设置x轴和y轴坐标范围, `x:0.5到1` `y:-1到1`
+ `clf` 清除画图中的内容
+ 'magic(5)' 创建 5 x 5矩阵, 每行每列和相等
+ `imagesc(A)` 将矩阵画成方格 不同的颜色对应矩阵中不同的值
+ `imagesc(magic(5)), colorbar, colormap gray;`  逗号分隔 连续调用函数.

## 循环控制语句

### for :

    for i = 1:10, # 遍历 1到10的数
           v(i) = 2^i
    end;

### while:

     while i <= 5,
       v(i) = 100;
       i = i + 1
     end;


### while true:

    while true,
        v(i) = 999;
        i = i+ 1;
         if i == 6,
            break;
         end;
    end;

### 定义函数和调用函数

    function y = squareThisNuber(x)
    y = x^2;

    squeareThisNumber(5) # 调用的时候需要和函数的文件在同一路径下

   `addpath('/User/jack4c/ml')` 添加目录,ocatve 可以调用该路径下的函数

## 向量化






