# setwd("~/Desktop/LCI plot_revised2/new plot/code")
library(gridExtra)
library(ggplot2)

# 读取第一个数据集并处理
C2_numeric <- read.csv("figure4a.csv", header = FALSE)
data <- data.frame(x = C2_numeric[, 1], y = C2_numeric[, 3] * 100)

# 创建第一个正方形图表
plot2 <- ggplot(data, aes(x = x, y = y)) + 
  geom_point(alpha = 0.25, size = 2, color = "blue") +
  theme_bw() +
  xlab("q") +
  ylab("Median Mean Absolute Percentage Error (%)") +
  ggtitle("5% missing: Percentage on test set with corresponding best parameters") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    aspect.ratio = 1  # 设置宽高比为1，使图表成为正方形
  )

# 读取第二个数据集并处理
values <- read.csv("figure4b.csv", header = FALSE)
seed_range <- 1:50
values <- as.numeric(values$V1) * 100
values_mean <- mean(values)
data2 <- data.frame(seed_range = seed_range, values = values)

# 创建第二个正方形图表
plot3 <- ggplot(data2, aes(x = seed_range, y = values)) + 
  geom_point(alpha = 0.25, size = 2, color = "blue") +
  geom_hline(yintercept = values_mean, color = "red", linetype = "dashed", size = 1) +
  theme_bw() +
  xlab("Random selection of 1200 processes") +
  ylab("Median Mean Absolute Percentage Error (%)") +
  ggtitle("5% missing: Percentage Error of 50 sub-testset, q=0.075 k=2") +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    aspect.ratio = 1  # 设置宽高比为1，使图表成为正方形
  )

# 将两个正方形图表并排显示
grid.arrange(plot2, plot3, ncol = 2, widths = c(1, 1))