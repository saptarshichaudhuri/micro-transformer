def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Prepare test prompts
    prompts = prepare_test_prompts(args)
    
    # Run text completion test
    completion_results = run_text_completion_test(model, tokenizer, prompts, args)
    
    # Run perplexity evaluation
    perplexity_results = run_perplexity_evaluation(model, tokenizer, prompts, args)
    
    # Analyze generation diversity
    diversity_results = analyze_generation_diversity(completion_results)
    
    # Create visualizations
    create_visualizations(completion_results, perplexity_results, diversity_results, args)
    
    # Save results
    save_test_results(completion_results, perplexity_results, diversity_results, args)

if __name__ == "__main__":
    main()