# test_utils.py
import os
import sys
import json
from datetime import datetime

# 添加项目路径到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import Get_file, scearch_content, Visualization

def test_file_processing_and_features():
    """测试文件处理和各项功能"""
    print("=== 文本信息检索系统功能测试 ===")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 1. 初始化文件处理器
    test_folder_path = r"E:\langurage_process\pro\test"
    print(f"1. 初始化文件处理器，目标路径: {test_folder_path}")
    
    try:
        # 检查路径是否存在
        if not os.path.exists(test_folder_path):
            print(f"错误: 测试文件夹不存在 - {test_folder_path}")
            return False
            
        # 初始化文件处理器（直接使用文件夹路径）
        file_processor = Get_file(test_folder_path, is_json=False)
        file_processor.init_read_dir()
        
        print(f"✓ 文件处理器初始化成功")
        print(f"  - 找到文件数量: {len(file_processor.file_name)}")
        print(f"  - 处理的文档总数: {file_processor.total_docs}")
        
    except Exception as e:
        print(f"✗ 文件处理器初始化失败: {str(e)}")
        return False
    
    # 2. 展示文件列表和内容预览
    print("\n2. 文件列表和内容预览:")
    for i, (file_path, file_name) in enumerate(zip(file_processor.file_path, file_processor.file_name)):
        print(f"  文件 {i+1}: {file_name}")
        content = file_processor.processed_files_content.get(file_path, "")
        if content and not content.startswith("错误:"):
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"    内容预览: {preview.replace(chr(10), ' ').replace(chr(13), ' ')}")
        else:
            print(f"    状态: {content}")
    
    # 3. 展示分词处理结果
    print("\n3. 分词处理结果展示:")
    sample_count = 0
    for file_path, content in file_processor.processed_files_content.items():
        if content and not content.startswith("错误:") and sample_count < 3:
            file_name = os.path.basename(file_path)
            print(f"\n  文件: {file_name}")
            
            # 展示原始内容片段
            original_sample = content[:200]
            print(f"    原始内容: {original_sample}...")
            
            # 进行分词处理
            processed_content = file_processor.process_text([content])
            if processed_content:
                processed_sample = processed_content[0][:200]
                words = processed_sample.split()[:20]  # 展示前20个词
                print(f"    分词结果: {' | '.join(words)}")
                print(f"    词汇数量: {len(processed_sample.split())}")
            
            sample_count += 1
    
    # 4. 词频统计展示
    print("\n4. 词频统计结果:")
    word_dict = file_processor.word_dict
    if word_dict:
        print(f"  总计高频词数量: {len(word_dict)}")
        print("  前20个高频词:")
        for i, (word, freq) in enumerate(list(word_dict.items())[:20]):
            print(f"    {i+1:2d}. {word} - {freq}次")
    else:
        print("  未生成词频统计")
    
    # 5. 倒排索引信息
    print("\n5. 倒排索引信息:")
    inverted_index = file_processor.inverted_index
    if inverted_index:
        print(f"  索引词汇总数: {len(inverted_index)}")
        print("  示例索引词条:")
        sample_words = list(inverted_index.keys())[:5]
        for word in sample_words:
            info = inverted_index[word]
            doc_count = info['doc_freq']
            print(f"    '{word}': 出现在 {doc_count} 个文档中")
    
    # 6. 测试搜索功能
    print("\n6. 搜索功能测试:")
    search_engine = scearch_content()
    
    # 选择测试查询词（从高频词中选择）
    test_queries = []
    if word_dict:
        test_queries = list(word_dict.keys())[:3]  # 取前3个高频词作为测试
    
    if not test_queries:
        test_queries = ["测试", "文档", "内容"]  # 默认测试词
    
    for query in test_queries:
        print(f"\n  测试查询词: '{query}'")
        
        # TF搜索
        tf_results = search_engine.tf_search(
            query, inverted_index, 
            file_processor.processed_files_content, 
            file_processor.file_word_count
        )
        print(f"    TF搜索: {tf_results['message']} (结果数: {tf_results.get('total_results', 0)})")
        
        # TF-IDF搜索
        tfidf_results = search_engine.tfidf_search(
            query, inverted_index,
            file_processor.processed_files_content,
            file_processor.file_word_count,
            file_processor.content_list
        )
        print(f"    TF-IDF搜索: {tfidf_results['message']} (结果数: {tfidf_results.get('total_results', 0)})")
        
        # 词向量搜索
        vector_results = search_engine.word_vector_search(
            query, inverted_index,
            file_processor.processed_files_content,
            file_processor.file_word_count,
            file_processor.content_list
        )
        print(f"    词向量搜索: {vector_results['message']} (结果数: {vector_results.get('total_results', 0)})")
        
        # 展示第一个搜索结果的详细信息
        if tf_results.get('results'):
            first_result = tf_results['results'][0]
            print(f"      最佳匹配: {os.path.basename(first_result['file_name'])}")
            print(f"      词频: {first_result['word_frequency']}, TF分数: {first_result['tf_score']}")
    
    # 7. 测试可视化功能
    print("\n7. 可视化功能测试:")
    visualizer = Visualization()
    
    # 确保输出目录存在
    visualizer.ensure_output_dir()
    
    if test_queries:
        test_query = test_queries[0]
        print(f"  使用查询词 '{test_query}' 测试可视化功能")
        
        # 词云图生成
        wordcloud_result = visualizer.generate_wordcloud(
            'test2', 
            file_processor.processed_files_content
        )
        print(f"    词云图生成: {wordcloud_result['message']}")
        if wordcloud_result['success']:
            print(f"      保存路径: {wordcloud_result['save_path']}")
            print(f"      相关文档数: {wordcloud_result.get('relevant_docs', 0)}")
        
        # 关键词排行图
        ranking_result = visualizer.generate_keyword_ranking(
            test_query,
            file_processor.processed_files_content,
            file_processor.file_word_count
        )
        print(f"    关键词排行图: {ranking_result['message']}")
        if ranking_result['success']:
            print(f"      保存路径: {ranking_result['save_path']}")
    
    # 文件后缀分析
    extension_result = visualizer.analyze_file_extensions(file_processor.file_name)
    print(f"    文件后缀分析: {extension_result['message']}")
    if extension_result['success']:
        print(f"      保存路径: {extension_result['save_path']}")
        print(f"      文件类型数: {extension_result.get('unique_extensions', 0)}")
    
    # 词频图表
    freq_chart_result = visualizer.create_word_frequency_chart(word_dict)
    print(f"    词频图表: {freq_chart_result['message']}")
    if freq_chart_result['success']:
        print(f"      保存路径: {freq_chart_result['save_path']}")
    
    # 8. 统计信息汇总
    print("\n8. 系统统计信息汇总:")
    print(f"  处理文件数量: {len(file_processor.file_name)}")
    print(f"  有效文档数量: {file_processor.total_docs}")
    print(f"  索引词汇数量: {len(inverted_index) if inverted_index else 0}")
    print(f"  高频词数量: {len(word_dict) if word_dict else 0}")
    
    # 文件类型统计
    if extension_result['success']:
        ext_stats = extension_result['extension_stats']
        print("  文件类型分布:")
        for stat in ext_stats[:5]:  # 显示前5种类型
            print(f"    {stat['extension']}: {stat['count']}个 ({stat['percentage']:.1f}%)")
    
    print("\n=== 测试完成 ===")
    return True

def test_specific_text_processing():
    """专门测试文本处理和分词功能"""
    print("\n=== 专项文本处理测试 ===")
    
    test_folder_path = r"E:\langurage_process\pro\test"
    
    try:
        file_processor = Get_file(test_folder_path, is_json=False)
        file_processor.init_read_dir()
        
        print("详细分词处理展示:")
        
        # 展示详细的分词过程
        for i, (file_path, content) in enumerate(file_processor.processed_files_content.items()):
            if content and not content.startswith("错误:") and i < 2:  # 只展示前2个文件
                file_name = os.path.basename(file_path)
                print(f"\n文件 {i+1}: {file_name}")
                print("-" * 40)
                
                # 原始内容
                original_sample = content[:300]
                print(f"原始内容片段:\n{original_sample}...")
                
                # 分词过程展示
                import jieba
                import re
                
                # 步骤1: 清理特殊字符
                cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', content)
                cleaned_sample = cleaned[:200]
                print(f"\n清理后内容:\n{cleaned_sample}...")
                
                # 步骤2: 分词
                words = list(jieba.cut(cleaned))
                words_sample = words[:30]  # 前30个词
                print(f"\n分词结果 (前30个):\n{' | '.join(words_sample)}")
                
                # 步骤3: 去停用词
                stopwords = file_processor._load_stopwords()
                filtered_words = [word.strip() for word in words 
                                if word.strip() and word.strip() not in stopwords and len(word.strip()) > 1]
                filtered_sample = filtered_words[:30]
                print(f"\n去停用词后 (前30个):\n{' | '.join(filtered_sample)}")
                
                # 统计信息
                print(f"\n统计信息:")
                print(f"  原始字符数: {len(content)}")
                print(f"  清理后字符数: {len(cleaned)}")
                print(f"  分词总数: {len(words)}")
                print(f"  过滤后词数: {len(filtered_words)}")
                print(f"  去重后词数: {len(set(filtered_words))}")
        
    except Exception as e:
        print(f"文本处理测试失败: {str(e)}")

if __name__ == "__main__":
    # 运行主要功能测试
    success = test_file_processing_and_features()
    
    if success:
        # 运行专项文本处理测试
        test_specific_text_processing()
    else:
        print("主要功能测试失败，跳过专项测试")
