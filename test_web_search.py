"""
Test script to verify duckduckgo-search is working
"""

def test_duckduckgo_import():
    """Test 1: Can we import the package?"""
    print("Test 1: Importing duckduckgo_search...")
    try:
        from duckduckgo_search import DDGS
        print(" SUCCESS: Package imported successfully!\n")
        return True
    except ImportError as e:
        print(f" FAILED: {e}")
        print("   Run: pip install duckduckgo-search==7.1.0\n")
        return False


def test_web_search():
    """Test 2: Can we actually search?"""
    print("Test 2: Testing web search...")
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text("best cooking oil for frying", max_results=2)
        
        if results:
            print(f" SUCCESS: Found {len(results)} results!")
            print("\nSample result:")
            print(f"  Title: {results[0]['title']}")
            print(f"  Snippet: {results[0]['body'][:100]}...\n")
            return True
        else:
            print("  WARNING: Search worked but returned no results\n")
            return False
            
    except Exception as e:
        print(f" FAILED: {e}\n")
        return False


def test_web_search_tool_format():
    """Test 3: Test the format used in MindDish tool"""
    print("Test 3: Testing MindDish tool format...")
    try:
        from duckduckgo_search import DDGS
        
        query = "substitute olive oil with butter"
        results = DDGS().text(query, max_results=3)
        
        # Format
        response = f" Web search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result['title']}\n   {result['body'][:150]}...\n\n"
        
        print(" SUCCESS: Tool format works!")
        print("\nFormatted output:")
        print(response[:300] + "...\n")
        return True
        
    except Exception as e:
        print(f" FAILED: {e}\n")
        return False


def main():
    print("="*60)
    print("MindDish.ai - DuckDuckGo Search Verification")
    print("="*60 + "\n")
    
    test1 = test_duckduckgo_import()
    
    if test1:
        test2 = test_web_search()
        test3 = test_web_search_tool_format()
        
        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Import Test:        {' PASS' if test1 else '❌ FAIL'}")
        print(f"Web Search Test:    {' PASS' if test2 else '❌ FAIL'}")
        print(f"Tool Format Test:   {' PASS' if test3 else '❌ FAIL'}")
        print("="*60 + "\n")
        
        if test1 and test2 and test3:
            print(" ALL TESTS PASSED!")
            print("Your web search is ready to use in MindDish.ai!\n")
        else:
            print("  Some tests failed. Check errors above.\n")
    else:
        print("Cannot proceed - package not installed.\n")


if __name__ == "__main__":
    main()