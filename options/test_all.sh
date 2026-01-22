#!/bin/bash

# 默认参数
INCLUDE_KEYWORD=""  # 默认包含所有文件
EXCLUDE_KEYWORDS=() # 默认不排除任何文件
GPUS=1

# 手动解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k)
            INCLUDE_KEYWORD="$2"
            shift 2
            ;;
        -e)
            shift
            # 收集所有排除关键词，直到遇到下一个选项或结束
            while [[ $# -gt 0 ]] && [[ "$1" != -* ]]; do
                EXCLUDE_KEYWORDS+=("$1")
                shift
            done
            ;;
        -g)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 遍历当前目录下所有 .yml 文件
found_files=0
for yml_file in "$SCRIPT_DIR"/*.yml; do
    if [[ ! -f "$yml_file" ]]; then
        continue  # 跳过非文件
    fi

    filename=$(basename "$yml_file")

    # 检查是否包含关键词（如果 -k 指定）
    if [[ -n "$INCLUDE_KEYWORD" ]] && [[ "$filename" != *"$INCLUDE_KEYWORD"* ]]; then
        continue  # 不匹配包含关键词，跳过
    fi

    # 检查是否匹配任何排除关键词
    skip_file=0
    for keyword in "${EXCLUDE_KEYWORDS[@]}"; do
        if [[ "$filename" == *"$keyword"* ]]; then
            echo "Skipping excluded file ($keyword): $yml_file"
            skip_file=1
            break
        fi
    done
    [[ $skip_file -eq 1 ]] && continue

    # 执行 Python 脚本
    echo "Running test with config: $yml_file"
    python Video-RetinexFormer/test.py --opt "$yml_file" -g "$GPUS"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to run $yml_file" >&2
        exit 1
    fi
    found_files=1
done

if [[ $found_files -eq 0 ]]; then
    echo "No matching YAML files found." >&2
    exit 1
fi

echo "All matching YAML configurations processed successfully."