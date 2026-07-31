# Contributing

1. `main` 分支只保存经过检查、能够运行的版本。
2. 三个人不得长期直接在 `main` 上开发。
3. 分支命名如下：

梁瀚升：

```text
feature/lhs-model
```

胡杨：

```text
feature/huyang-data-eval
```

朱福康：

```text
feature/zhufukang-engineering
```

4. 每完成一个小功能就提交一次，不要积累成一个巨大提交。
5. 禁止提交：

- 完整数据集
- 大模型权重
- 虚拟环境
- 密钥
- `.env`
- 大型日志

6. 合并到 `main` 前，需要说明：

- 修改了什么
- 如何运行
- 是否实际测试
- 运行结果

7. 推荐提交信息：

```text
feat(model): ...
feat(data): ...
feat(eval): ...
feat(engineering): ...
fix: ...
docs: ...
chore: ...
```
