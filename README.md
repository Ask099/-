项目功能说明：
这是一个厄尔尼诺现象预测系统
它接受最短为24个月的sst,ht,nino3.4数据作为输入，输出未来三个月的nino3.4指数，并对预测结果简要分析

项目架构：
项目基于python语言编写
预测模块是基于pytorch的lstm网络
交互模块使用流水线模式：启动程序->在本地端口通过浏览器打开一个界面->输入数据后，程序执行预测返回数据->前端界面展示结果
其中前端界面基于streamlit

如何使用：
1.启动Anaconda prompt，转换到本项目对应文件夹
2.在prompt中执行命令 conda env create -f environment.yml
3.待环境安装成功后，激活环境: conda activate enso-prediction
4.激活后，运行启动程序: python app.py 启动后按指引上传数据预测即可