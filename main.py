import streamlit as st
import ollama
import json
import re
from crewai import Agent, Task, Crew, Process
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from textwrap import dedent
from enum import Enum

# Supported languages enum
class ProgrammingLanguage(Enum):
    PYTHON = "Python"
    GO = "Go"
    RUST = "Rust"
    C = "C"
    CPP = "C++"
    CSHARP = "C#"
    JAVASCRIPT = "JavaScript"
    TYPESCRIPT = "TypeScript"
    JAVA = "Java"
    KOTLIN = "Kotlin"
    SWIFT = "Swift"

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = "codellama:python"
if 'language' not in st.session_state:
    st.session_state.language = ProgrammingLanguage.PYTHON.value
if 'available_models' not in st.session_state:
    # Expanded model list with specialized coding models
    st.session_state.available_models = [
        # Python-specialized
        "codellama:python", "deepseek-coder:6.7b", "phi3:3.8b", "stable-code",
        # Multi-language capable
        "codellama:70b-instruct", "codellama:34b-instruct", "mistral:7b-instruct",
        "llama3:70b-instruct", "llama3:8b-instruct", "mixtral:8x7b-instruct",
        # Systems programming focused
        "deepseek-coder:1.3b-base", "starcoder2:3b", "starcoder2:7b", "starcoder2:15b",
        # Enterprise/backend focused
        "wizardcoder:15b", "phind-codellama:34b", "magicoder:7b"
    ]
if 'code_input' not in st.session_state:
    st.session_state.code_input = ""
if 'review_results' not in st.session_state:
    st.session_state.review_results = None
if 'teach_results' not in st.session_state:
    st.session_state.teach_results = None
if 'generate_results' not in st.session_state:
    st.session_state.generate_results = None

# Configure page
st.set_page_config(page_title="Code Crew", layout="wide", page_icon="🌐")
st.title("🌐 Code Crew: Multi-Language AI Assistant")

# Language-specific model recommendations
LANG_MODEL_CATEGORIES = {
    "Python": {
        "category": "🐍 Python Specialized",
        "models": ["codellama:python", "deepseek-coder:6.7b", "phi3:3.8b", "stable-code"]
    },
    "Go": {
        "category": "🐹 Go Optimized",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "mistral:7b-instruct"]
    },
    "Rust": {
        "category": "🦀 Rust Focused",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "wizardcoder:15b"]
    },
    "C": {
        "category": "⚙️ C Systems",
        "models": ["codellama:70b-instruct", "deepseek-coder:1.3b-base", "starcoder2:15b", "phind-codellama:34b"]
    },
    "C++": {
        "category": "🔫 C++ Performance",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "wizardcoder:15b", "phind-codellama:34b"]
    },
    "C#": {
        "category": ".NET Ecosystem",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "phi3:3.8b", "magicoder:7b"]
    },
    "JavaScript": {
        "category": "🌐 Web Development",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "mistral:7b-instruct"]
    },
    "TypeScript": {
        "category": "🟦 TypeScript",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "phi3:3.8b"]
    },
    "Java": {
        "category": "☕ Java Enterprise",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "wizardcoder:15b"]
    },
    "Kotlin": {
        "category": "🤖 Kotlin/Android",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "phi3:3.8b"]
    },
    "Swift": {
        "category": "🍏 Swift/iOS",
        "models": ["codellama:70b-instruct", "deepseek-coder:6.7b", "starcoder2:15b", "phi3:3.8b"]
    }
}

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Language selection with categories
    lang_options = [lang.value for lang in ProgrammingLanguage]
    lang_categories = {
        "Web": ["JavaScript", "TypeScript"],
        "Systems": ["Rust", "C", "C++", "Go"],
        "Enterprise": ["Java", "C#", "Kotlin"],
        "Mobile": ["Swift", "Kotlin"],
        "Scripting": ["Python"]
    }
    
    # Create categorized language selection
    category = st.selectbox("Language Category", list(lang_categories.keys()), index=0)
    filtered_langs = lang_categories[category]
    selected_lang = st.selectbox(
        "Primary Language",
        [lang for lang in lang_options if lang in filtered_langs],
        index=0
    )
    st.session_state.language = selected_lang
    
    # Model selection with language-aware categories
    lang_config = LANG_MODEL_CATEGORIES.get(st.session_state.language, {
        "category": "🌍 General Purpose",
        "models": ["codellama:70b-instruct", "mistral:7b-instruct", "llama3:70b-instruct"]
    })
    
    st.subheader(f"🤖 {lang_config['category']}")
    
    # Get recommended models and fallback to available
    recommended_models = lang_config["models"]
    other_models = [m for m in st.session_state.available_models if m not in recommended_models]
    all_models = recommended_models + other_models
    
    st.session_state.model = st.selectbox(
        "Ollama Model",
        all_models,
        index=0,
        help=f"Specialized models for {st.session_state.language} development"
    )
    
    # Model status check with caching - FIXED KEY ERROR
    @st.cache_data(ttl=3600)
    def get_model_status(model_name):
        try:
            # Get model information
            model_info = ollama.show(model_name)
            
            # Extract size information safely
            size_bytes = None
            param_size = "Unknown"
            
            # Check different possible locations for size information
            if "details" in model_info:
                details = model_info["details"]
                # Try different possible size fields
                size_bytes = details.get("size") or details.get("file_size") or details.get("model_size")
                param_size = details.get("parameter_size", "Unknown")
            
            # Convert to GB if available
            size_info = ""
            if size_bytes:
                try:
                    size_gb = size_bytes / (1024**3)
                    size_info = f"{size_gb:.1f}GB"
                except (TypeError, ValueError):
                    size_info = "Unknown size"
            else:
                size_info = "Size unknown"
            
            return True, None, size_info, param_size
        except Exception as e:
            return False, str(e), None, None
    
    available, error_msg, size_info, param_size = get_model_status(st.session_state.model)
    
    if available:
        st.success(f"✅ {st.session_state.model} ready")
        # Only show details if we have them
        if size_info and param_size:
            st.caption(f"💡 {size_info} • Parameters: {param_size}")
        elif param_size != "Unknown":
            st.caption(f"💡 Parameters: {param_size}")
    else:
        st.warning(f"⚠️ Model not available: {error_msg}")
        st.info(f"Install with: `ollama pull {st.session_state.model}`")
    
    # Language-specific resources
    st.markdown("---")
    st.subheader(f"📚 {st.session_state.language} Resources")
    lang_resources = {
        "Python": ["PEP 8 Guide", "Python Docs", "Real Python Tutorials"],
        "Go": ["Effective Go", "Go by Example", "Go Proverbs"],
        "Rust": ["Rust Book", "Rustlings", "Rust API Guidelines"],
        "C": ["C Standard (C17)", "Modern C", "C Reference"],
        "C++": ["C++ Core Guidelines", "CppReference", "Effective Modern C++"],
        "C#": [".NET Docs", "C# Coding Conventions", "Design Guidelines"],
        "JavaScript": ["MDN Web Docs", "JavaScript.info", "You Don't Know JS"],
        "TypeScript": ["TS Handbook", "TypeScript Deep Dive", "TS Best Practices"],
        "Java": ["Java Docs", "Effective Java", "Java Design Patterns"],
        "Kotlin": ["Kotlin Docs", "Kotlin Idioms", "Android Dev Guides"],
        "Swift": ["Swift.org", "Swift API Design Guidelines", "Apple Dev Tutorials"]
    }
    
    resources = lang_resources.get(st.session_state.language, [])
    for i, resource in enumerate(resources[:3], 1):
        st.markdown(f"{i}. [{resource}](https://example.com/{resource.replace(' ', '-')})")
    
    if len(resources) > 3:
        st.caption(f"+{len(resources)-3} more resources available")
    
    st.markdown("---")
    st.caption("💡 **Pro Tip**: Select specialized models for better code understanding in your language!")

# Initialize Ollama LLM
@st.cache_resource
def get_llm(model_name):
    """Initialize Ollama LLM with optimized settings"""
    return OllamaLLM(
        model=model_name,
        temperature=0.2,
        num_predict=1024,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"]
    )

try:
    llm = get_llm(st.session_state.model)
except Exception as e:
    st.error(f"🚨 LLM initialization failed: {str(e)}")
    st.info("1. Ensure Ollama is running (`ollama serve`)\n2. Install required package: `pip install -U langchain-ollama`")
    st.stop()

# Language-specific agent specializations
def create_language_specialist(language):
    """Creates language-specific agent backstories and goals"""
    specializations = {
        "Python": (
            "You are a Principal Python Engineer at a FAANG company. You've authored popular PyPI packages and contributed to CPython. "
            "You obsess over PEP-8 compliance, type hinting, and writing truly Pythonic code that leverages the standard library effectively.",
            "Write elegant, idiomatic Python with comprehensive type hints, docstrings, and context manager usage where appropriate"
        ),
        "Go": (
            "You are a Staff Engineer at Google who helped design Go's concurrency patterns. You've contributed to the Go standard library "
            "and specialize in writing highly concurrent, garbage-collector friendly code that follows Go's idiomatic patterns.",
            "Write clean Go code using goroutines/channels appropriately, proper error handling, and Go's interface-based design"
        ),
        "Rust": (
            "You are a Rust core team member who helped design the borrow checker. You write zero-cost abstractions with perfect memory safety "
            "and leverage Rust's type system to eliminate runtime errors before they happen.",
            "Write safe, performant Rust with proper ownership, lifetimes, and error handling using anyhow/thiserror"
        ),
        "C": (
            "You are a Linux kernel maintainer with 20+ years of C experience. You write portable, secure C code that works on everything from "
            "microcontrollers to supercomputers, with meticulous attention to memory safety and undefined behavior prevention.",
            "Write secure C code with proper memory management, bounds checking, and platform-agnostic practices"
        ),
        "C++": (
            "You are a C++ Standards Committee member specializing in C++20/23 features. You write modern C++ using RAII, smart pointers, "
            "and templates while avoiding legacy patterns and raw pointer pitfalls.",
            "Write modern C++ using STL, smart pointers, concepts, and ranges with clear ownership semantics"
        ),
        "C#": (
            "You are a Principal .NET Architect at Microsoft who helped design C# 10+ features. You specialize in high-performance .NET code "
            "with async/await patterns, memory-efficient designs, and proper dependency injection.",
            "Write idiomatic C# with async/await, proper DI patterns, and performance-conscious allocations"
        ),
        "JavaScript": (
            "You are a TC39 committee member who helped shape modern JavaScript. You write performant, maintainable code leveraging the latest "
            "ECMAScript features while maintaining compatibility across environments.",
            "Write modern JavaScript using ES2022+ features, proper async patterns, and module organization"
        ),
        "TypeScript": (
            "You are a TypeScript core team contributor who specializes in advanced type systems. You write type-safe code that leverages "
            "TypeScript's full capabilities while maintaining excellent developer experience and runtime performance.",
            "Write type-safe TypeScript with advanced generics, utility types, and strict null checking"
        ),
        "Java": (
            "You are a Java Champion who has contributed to the JDK and popular frameworks. You specialize in writing high-performance, "
            "thread-safe Java code that leverages modern features while maintaining backwards compatibility.",
            "Write modern Java using records, streams, and modules with proper concurrency patterns"
        ),
        "Kotlin": (
            "You are a JetBrains engineer who helped design Kotlin's coroutines. You write idiomatic Kotlin that leverages extension functions, "
            "null safety, and coroutine patterns while maintaining interoperability with Java ecosystems.",
            "Write concise Kotlin with coroutines, extension functions, and sealed classes"
        ),
        "Swift": (
            "You are an Apple Frameworks engineer who specializes in Swift performance optimization. You write safe, expressive Swift code "
            "that leverages value types, protocol-oriented design, and Swift's concurrency model effectively.",
            "Write idiomatic Swift with async/await, value semantics, and protocol-oriented design"
        )
    }
    
    # Fallback to Python specialization for unsupported languages
    return specializations.get(language, specializations["Python"])

# Create CrewAI agents with language specialization
def create_crew(task_type, user_input, language):
    # Common tools
    search_tool = DuckDuckGoSearchRun()
    lang_backstory, lang_goal = create_language_specialist(language)
    
    # Common role definitions with language context
    educator_backstory = f"""You are a world-class {language} educator with 15+ years of experience. 
    You've taught at top universities and created best-selling courses on {language} development. 
    You excel at explaining complex {language} concepts using real-world analogies and practical examples."""
    
    bug_hunter_backstory = f"""{lang_backstory} You've prevented critical production outages in {language} systems 
    by finding subtle memory leaks, race conditions, and security vulnerabilities that others missed."""
    
    code_gardener_backstory = f"""{lang_backstory} You refactored the legacy {language} codebase at a Fortune 500 company 
    into a maintainable modern system that reduced bug reports by 70%. You obsess over readability and maintainability."""
    
    architect_backstory = f"""{lang_backstory} You designed the {language} architecture for a system handling 
    1M+ requests/sec. You balance performance, scalability, and developer experience in every decision."""
    
    # Language-agnostic agents
    code_professor = Agent(
        role=f"{language} Educator",
        goal=f"Explain {language} concepts clearly with practical examples",
        backstory=educator_backstory,
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[search_tool]
    )
    
    # Language-specialized agents
    bug_hunter = Agent(
        role=f"{language} Quality Engineer",
        goal=f"Identify bugs and security issues in {language} code",
        backstory=bug_hunter_backstory,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    code_gardener = Agent(
        role=f"{language} Code Quality Specialist",
        goal=f"Improve {language} code quality and maintainability",
        backstory=code_gardener_backstory,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    architect = Agent(
        role=f"{language} Software Architect",
        goal=f"Design robust {language} solutions",
        backstory=architect_backstory,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    # Create tasks based on type and language
    if task_type == "review":
        review_task = Task(
            description=f"""Analyze this {language} code for:
            1. Language-specific bugs and anti-patterns
            2. Memory management issues ({'pointers/gc' if language in ['C', 'C++', 'Go'] else 'garbage collection'})
            3. Concurrency/thread safety issues
            4. Security vulnerabilities specific to {language}
            5. Violations of {language} style guides (e.g., {'PEP-8' if language=='Python' else 'Rust API Guidelines' if language=='Rust' else 'Effective Go'})
            
            Code to review:
            ```{language.lower()}
            {user_input}
            ```
            
            Provide line-by-line feedback with {language}-specific best practices. Be concise and actionable.""",
            agent=bug_hunter,
            expected_output="Detailed bug report with line numbers and fixes"
        )
        
        refactor_task = Task(
            description=f"""Refactor this {language} code while preserving functionality:
            ```{language.lower()}
            {user_input}
            ```
            
            Requirements:
            - Follow {language} style guides and idioms (e.g., {'PEP-8' if language=='Python' else 'Rust API Guidelines' if language=='Rust' else 'Effective Go'})
            - Improve readability and maintainability
            - Add proper error handling using {language}-idiomatic approaches
            - Include documentation/comments appropriate for {language}
            - Optimize performance where critical
            
            Output ONLY the improved code without explanations. Use {language} best practices.""",
            agent=code_gardener,
            expected_output="Refactored code with improved quality"
        )
        
        return Crew(
            agents=[bug_hunter, code_gardener],
            tasks=[review_task, refactor_task],
            verbose=2,
            process=Process.sequential
        )
    
    elif task_type == "teach":
        teaching_task = Task(
            description=f"""Explain this {language} concept clearly:
            {user_input}
            
            Structure your explanation:
            1. Core concept with {language}-specific context
            2. Comparison to similar concepts in other languages (if helpful)
            3. Practical {language} example with comments
            4. Common {language}-specific pitfalls
            5. Official {language} documentation references""",
            agent=code_professor,
            expected_output="Comprehensive educational explanation with examples"
        )
        
        return Crew(
            agents=[code_professor],
            tasks=[teaching_task],
            verbose=2
        )
    
    elif task_type == "generate":
        generation_task = Task(
            description=f"""Generate production-ready {language} code for:
            {user_input}
            
            Requirements:
            - Follow {language} best practices and idioms
            - Include comprehensive error handling using {language}'s standard approaches
            - Add documentation appropriate for {language} (docstrings, /// comments, etc.)
            - Consider performance characteristics of {language} runtime
            - Handle edge cases specific to {language}
            - Use modern {language} features where appropriate
            
            {lang_goal}""",
            agent=architect,
            expected_output="Complete, production-ready code implementation"
        )
        
        review_task = Task(
            description=f"""Review the generated {language} code for:
            1. Language-specific anti-patterns
            2. Memory safety issues
            3. Concurrency correctness
            4. Security vulnerabilities in {language} context
            5. Adherence to {language} community standards
            
            Code to review:
            ```{language.lower()}
            {{previous_task_output}}
            ```
            
            Flag any issues but DO NOT rewrite the code. Provide concise feedback only.""",
            agent=bug_hunter,
            expected_output="Security and quality review report"
        )
        
        return Crew(
            agents=[architect, bug_hunter],
            tasks=[generation_task, review_task],
            verbose=2,
            process=Process.sequential
        )

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["🔍 Code Review", "🎓 Code Teaching", "✨ Code Generation"])

# Tab 1: Code Review (Multi-Language)
with tab1:
    st.header(f"🐞 {st.session_state.language} Code Review")
    st.caption(f"Get expert analysis for your {st.session_state.language} code")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        language = st.selectbox(
            "Code Language",
            [lang.value for lang in ProgrammingLanguage],
            index=[lang.value for lang in ProgrammingLanguage].index(st.session_state.language),
            key="review_lang"
        )
    
    with col2:
        sample_type = st.selectbox(
            "Load Example",
            ["None", "Memory Leak", "Concurrency Bug", "Security Flaw", "Idiomatic Refactor", "Performance Issue"],
            key="review_sample"
        )
    
    # Sample codes for different languages
    sample_codes = {
        "Python": {
            "Memory Leak": dedent('''\
                def process_data():
                    cache = {}
                    while True:
                        data = [i for i in range(10000)]
                        cache[len(cache)] = data  # Memory leak
            '''),
            "Concurrency Bug": dedent('''\
                import threading
                
                counter = 0
                
                def increment():
                    global counter
                    for _ in range(100000):
                        counter += 1  # Race condition
                
                threads = [threading.Thread(target=increment) for _ in range(5)]
                for t in threads: t.start()
                for t in threads: t.join()
                print(counter)
            '''),
            "Performance Issue": dedent('''\
                def calculate_factors(n):
                    factors = []
                    for i in range(1, n+1):
                        if n % i == 0:
                            factors.append(i)
                    return factors
                
                # Inefficient for large numbers
                print(calculate_factors(1000000))
            ''')
        },
        "Go": {
            "Concurrency Bug": dedent('''\
                package main
                
                import "fmt"
                
                func main() {
                    ch := make(chan int)
                    go func() {
                        ch <- 42
                        close(ch)
                    }()
                    fmt.Println(<-ch)
                    fmt.Println(<-ch) // Blocked read on closed channel
                }
            '''),
            "Memory Leak": dedent('''\
                package main
                
                var cache = make(map[int][]byte)
                
                func processData(id int, data []byte) {
                    cache[id] = data // Memory leak - never cleaned
                }
            ''')
        },
        "Rust": {
            "Memory Safety": dedent('''\
                fn main() {
                    let mut v = vec![1, 2, 3];
                    let x = &v[0];
                    v.push(4); // Invalidates reference
                    println!("{}", x);
                }
            '''),
            "Concurrency": dedent('''\
                use std::thread;
                
                fn main() {
                    let mut data = vec![1, 2, 3];
                    
                    thread::spawn(move || {
                        data.push(4); // Ownership violation
                    });
                    
                    println!("{:?}", data);
                }
            ''')
        },
        "JavaScript": {
            "Memory Leak": dedent('''\
                function setupListeners() {
                    const elements = document.querySelectorAll('.item');
                    elements.forEach(el => {
                        el.addEventListener('click', () => {
                            console.log('Clicked');
                        });
                    });
                    // Memory leak: listeners not removed on DOM changes
                }
            '''),
            "Async Bug": dedent('''\
                async function fetchData() {
                    try {
                        const response = await fetch('/api/data');
                        const data = response.json();
                        return data; // Forgot to await json()
                    } catch (error) {
                        console.error('Failed:', error);
                    }
                }
            ''')
        }
    }
    
    # Load sample code
    if sample_type != "None":
        code_samples = sample_codes.get(language, {})
        st.session_state.code_input = code_samples.get(sample_type, "")
    
    code_input = st.text_area(
        "Your Code",
        value=st.session_state.code_input,
        height=300,
        placeholder=f"Paste your {language} code here...",
        key="review_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        review_btn = st.button("🚀 Analyze Code", type="primary", use_container_width=True)
    with col2:
        if st.button("🧹 Clear Code", use_container_width=True):
            st.session_state.code_input = ""
            st.rerun()
    
    if review_btn and code_input.strip():
        with st.spinner(f"🧠 {language} experts analyzing your code..."):
            try:
                crew = create_crew("review", code_input, language)
                result = crew.kickoff()
                
                # Improved result parsing
                result_str = str(result)
                if "-----" in result_str:
                    parts = result_str.split("-----", 1)
                    analysis = parts[0].strip()
                    fixed_code = parts[1].strip() if len(parts) > 1 else code_input
                else:
                    analysis = "Analysis completed successfully"
                    fixed_code = result_str
                
                st.session_state.review_results = {
                    "analysis": analysis,
                    "fixed_code": fixed_code,
                    "language": language
                }
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Try simplifying your code or switching to a more capable model")
    
    if st.session_state.review_results and st.session_state.review_results.get("language") == language:
        st.subheader("🔎 Expert Analysis")
        
        analysis_col, fixed_col = st.columns([1, 1])
        
        with analysis_col:
            st.markdown("### 📋 Analysis Report")
            st.markdown(st.session_state.review_results["analysis"])
        
        with fixed_col:
            st.markdown("### ✅ Improved Code")
            lang_syntax = language.lower()
            if language == "C++": lang_syntax = "cpp"
            elif language == "C#": lang_syntax = "csharp"
            elif language == "TypeScript": lang_syntax = "typescript"
            
            st.code(st.session_state.review_results["fixed_code"], language=lang_syntax)
            st.copy_button("📋 Copy Fixed Code", st.session_state.review_results["fixed_code"], key="copy_review")

# Tab 2: Code Teaching (Multi-Language)
with tab2:
    st.header(f"🎓 {st.session_state.language} Concepts Explained")
    st.caption(f"Ask about any {st.session_state.language} concept or pattern")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        language = st.selectbox(
            "Concept Language",
            [lang.value for lang in ProgrammingLanguage],
            index=[lang.value for lang in ProgrammingLanguage].index(st.session_state.language),
            key="teach_lang"
        )
    
    with col2:
        concept_type = st.selectbox(
            "Concept Type",
            ["Language Fundamentals", "Concurrency", "Memory Management", "Design Patterns", "Best Practices", "Ecosystem Tools"],
            key="concept_type"
        )
    
    concept_input = st.text_area(
        "Concept or Code to Explain",
        height=150,
        placeholder=f"Explain {('goroutines' if language=='Go' else 'lifetimes' if language=='Rust' else 'async/await' if language in ['JavaScript','TypeScript','C#'] else 'smart pointers')} in {language}...",
        key="teach_input"
    )
    
    teach_btn = st.button("👩‍🏫 Get Explanation", type="primary")
    
    if teach_btn and concept_input.strip():
        with st.spinner(f"🧠 {language} professor preparing lesson..."):
            try:
                crew = create_crew("teach", concept_input, language)
                result = crew.kickoff()
                st.session_state.teach_results = str(result)
            except Exception as e:
                st.error(f"Failed to generate explanation: {str(e)}")
                st.info("Try a more specific question or switch to a larger model")
    
    if st.session_state.teach_results:
        st.subheader("📚 Expert Explanation")
        st.markdown(st.session_state.teach_results)

# Tab 3: Code Generation (Multi-Language)
with tab3:
    st.header(f"✨ {st.session_state.language} Code Generator")
    st.caption(f"Describe what you want to build in {st.session_state.language}")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        language = st.selectbox(
            "Target Language",
            [lang.value for lang in ProgrammingLanguage],
            index=[lang.value for lang in ProgrammingLanguage].index(st.session_state.language),
            key="gen_lang"
        )
    
    with col2:
        framework_ecosystem = st.selectbox(
            "Framework/Ecosystem",
            ["Standard Library", "Web Framework", "Data Science", "Systems Programming", "Mobile", "Cloud Native"],
            key="framework"
        )
    
    gen_prompt = st.text_area(
        "Feature Description",
        height=150,
        placeholder=f"Create a {('REST API with FastAPI' if language=='Python' else 'concurrent web crawler' if language=='Go' else 'memory-safe parser' if language=='Rust' else 'React component with TypeScript hooks')}...",
        key="gen_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        complexity = st.select_slider(
            "Complexity Level",
            options=["Simple", "Intermediate", "Production-Ready"],
            value="Intermediate"
        )
    with col2:
        include_tests = st.checkbox("Include Unit Tests", value=True)
    
    gen_btn = st.button("🚀 Generate Code", type="primary")
    
    if gen_btn and gen_prompt.strip():
        # Enhance prompt with language context
        ecosystem_details = {
            "Python": {"Web Framework": "FastAPI/Flask", "Data Science": "Pandas/NumPy", "Cloud Native": "Django/AWS"},
            "JavaScript": {"Web Framework": "React/Next.js", "Cloud Native": "Node.js/Express"},
            "TypeScript": {"Web Framework": "React/Angular", "Cloud Native": "NestJS/AWS CDK"},
            "Go": {"Web Framework": "Gin/Echo", "Cloud Native": "Kubernetes operators", "Systems Programming": "CLI tools"},
            "Rust": {"Systems Programming": "async-std/tokio", "Web Assembly": "wasm-bindgen", "Embedded": "no_std"},
            "C#": {"Web Framework": "ASP.NET Core", "Mobile": "Xamarin/MAUI", "Cloud Native": "Azure Functions"}
        }.get(language, {})
        
        ecosystem = ecosystem_details.get(framework_ecosystem, framework_ecosystem)
        
        full_prompt = f"""
        Language: {language}
        Ecosystem: {ecosystem}
        Complexity: {complexity}
        {"Include unit tests with framework-appropriate testing libraries" if include_tests else "No tests needed"}
        
        Task: {gen_prompt}
        
        Requirements:
        - Use modern {language} idioms and best practices
        - Apply appropriate design patterns for this use case
        - Handle errors properly using {language}'s error handling mechanisms
        - Include documentation comments where critical
        """
        
        with st.spinner(f"🧠 {language} architects designing your solution..."):
            try:
                crew = create_crew("generate", full_prompt, language)
                result = crew.kickoff()
                st.session_state.generate_results = {
                    "code": str(result),
                    "language": language
                }
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                st.info("Try a simpler request or switch to a more capable model")
    
    if st.session_state.generate_results and st.session_state.generate_results.get("language") == language:
        st.subheader("💻 Generated Code")
        
        lang_syntax = language.lower()
        if language == "C++": lang_syntax = "cpp"
        elif language == "C#": lang_syntax = "csharp"
        elif language == "TypeScript": lang_syntax = "typescript"
        
        st.code(st.session_state.generate_results["code"], language=lang_syntax)
        
        # Copy button with unique key
        st.copy_button(
            "📋 Copy Generated Code", 
            st.session_state.generate_results["code"],
            key="copy_generate"
        )

# Footer with language stats and model info
st.markdown("---")
lang_stats = {
    "Python": "Dominant in AI/ML and data science • ~25% of all GitHub commits",
    "Go": "Cloud infrastructure standard • Kubernetes/Docker written in Go",
    "Rust": "Most loved language 8 years running • Memory safety without GC",
    "JavaScript": "Language of the web • 98% of websites use JS",
    "TypeScript": "Fastest growing language • Adopted by Google, Microsoft, Airbnb",
    "C#": ".NET ecosystem leader • Game development with Unity",
    "Java": "Enterprise backend standard • 3 billion devices run Java",
    "Kotlin": "Official Android language • 100% interoperable with Java",
    "Swift": "Apple ecosystem standard • Memory safe with modern syntax",
    "C": "Systems programming foundation • Powers operating systems and embedded devices",
    "C++": "High-performance powerhouse • Widely used in AAA games, finance, and real-time engines"
}

st.caption(f"💡 **{st.session_state.language} Insight**: {lang_stats.get(st.session_state.language, 'General purpose programming')}")
model_size = "7B-70B" if any(x in st.session_state.model for x in ["70b", "34b"]) else "1B-13B"
st.caption(f"🧠 **Model Capability**: {st.session_state.model} ({model_size} parameters) • All processing happens locally")
st.caption("🧑‍🚀 Powered by CrewAI multi-agent system • Refresh page to reset session")