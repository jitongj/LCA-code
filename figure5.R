library(ggplot2)
library(gridExtra)

# setwd("~/Desktop/LCI plot_revised2/new plot/code")

# 读取第一个数据集并处理
scatter <- read.csv("Frequency_5%_new.csv", header = TRUE)
scatter <- scatter[complete.cases(scatter), ]
scatter$frequency <- log10(abs(scatter$frequency))
scatter$percentage.error <- scatter$percentage.error * 100  # 将 y 轴坐标扩大 100 倍

# 创建第一个图表
plot1 <- ggplot(scatter, aes(x = frequency, y = percentage.error)) + 
  geom_point(alpha = 0.25, size = 1, color = "blue") +  # 点的颜色设置为蓝色
  theme_bw() + 
  geom_smooth(aes(group = 1), method = "lm", se = FALSE, linetype = "dashed", fullrange = FALSE) +
  xlab("Frequency of Flow (log10)") + 
  ylab("Flow's Median Mean Absolute Percentage Error (%)") +  # 更新 y 轴标签
  scale_x_continuous(limits = c(0, max(scatter$frequency, na.rm = TRUE))) +
  scale_y_continuous(limits = c(0, 100)) +  # 更新 y 轴范围
  ggtitle("5% missing flows") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# 读取第二个数据集并处理
scatter1 <- read.csv("5%missing_new.csv", header = TRUE)
scatter1[scatter1$percentage.error > 1, ]$percentage.error <- 1
scatter1$percentage.error <- scatter1$percentage.error * 100  # 将 y 轴坐标扩大 100 倍

# 创建第二个图表
plot2 <- ggplot(scatter1, aes(x = actual.data, y = percentage.error)) +
  geom_point(alpha = 0.25, size = 1, color = "blue") +  # 点的颜色设置为蓝色
  theme_bw() +
  xlab("True Value") +
  ylab("Absolute Percentage Error (%)") +  # 更新 y 轴标签
  ggtitle("5% missing flows") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# 将两个图表并排显示
grid.arrange(plot1, plot2, ncol = 2, widths = c(1, 1))

