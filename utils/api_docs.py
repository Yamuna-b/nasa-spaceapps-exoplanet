"""
API documentation system for the Exoplanet AI application.
"""
import streamlit as st
from typing import Dict, List, Any
import json

class APIDocumentation:
    """Comprehensive API documentation system."""
    
    def __init__(self):
        self.api_endpoints = self._load_api_endpoints()
        self.api_examples = self._load_api_examples()
    
    def show_api_docs(self):
        """Show the main API documentation page."""
        st.header("🔌 API Documentation")
        
        st.markdown("""
        The Exoplanet AI API provides programmatic access to our machine learning models 
        and exoplanet detection capabilities. This RESTful API allows you to integrate 
        exoplanet analysis into your own applications and workflows.
        """)
        
        # API overview
        self.show_api_overview()
        
        # Navigation
        doc_section = st.selectbox(
            "Choose a section:",
            [
                "Getting Started",
                "Authentication",
                "Endpoints Reference",
                "Data Models",
                "Code Examples",
                "Rate Limits & Quotas",
                "Error Handling",
                "SDKs & Libraries"
            ],
            key="api_doc_section"
        )
        
        if doc_section == "Getting Started":
            self.show_getting_started()
        elif doc_section == "Authentication":
            self.show_authentication_docs()
        elif doc_section == "Endpoints Reference":
            self.show_endpoints_reference()
        elif doc_section == "Data Models":
            self.show_data_models()
        elif doc_section == "Code Examples":
            self.show_code_examples()
        elif doc_section == "Rate Limits & Quotas":
            self.show_rate_limits()
        elif doc_section == "Error Handling":
            self.show_error_handling()
        else:
            self.show_sdks_libraries()
    
    def show_api_overview(self):
        """Show API overview and key features."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Version", "v1.0")
        
        with col2:
            st.metric("Base URL", "https://api.exoplanet-ai.com")
        
        with col3:
            st.metric("Response Format", "JSON")
        
        st.markdown("""
        ### Key Features
        - **Machine Learning Predictions**: Get exoplanet classifications for your data
        - **Multiple Models**: Access Random Forest, Logistic Regression, and SVM models
        - **Batch Processing**: Analyze multiple candidates in a single request
        - **Real-time Results**: Get predictions in seconds
        - **Confidence Scores**: Understand prediction reliability
        - **Feature Analysis**: Get feature importance and explanations
        """)
    
    def show_getting_started(self):
        """Show getting started guide."""
        st.subheader("🚀 Getting Started")
        
        st.markdown("""
        ### 1. Get Your API Key
        
        First, you'll need an API key to authenticate your requests:
        
        1. Sign up for an account at [exoplanet-ai.com](https://exoplanet-ai.com)
        2. Go to your Profile → API Keys
        3. Generate a new API key
        4. Copy and securely store your key
        """)
        
        st.code("""
        # Your API key will look like this:
        API_KEY = "exa_1234567890abcdef1234567890abcdef"
        """, language="python")
        
        st.markdown("""
        ### 2. Make Your First Request
        
        Here's a simple example to get you started:
        """)
        
        st.code("""
        import requests
        
        # API configuration
        API_KEY = "your_api_key_here"
        BASE_URL = "https://api.exoplanet-ai.com/v1"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Sample data
        data = {
            "dataset": "kepler",
            "model": "rf",
            "candidates": [
                {
                    "koi_period": 10.5,
                    "koi_prad": 1.2,
                    "koi_depth": 100,
                    "koi_duration": 2.5,
                    "koi_ingress": 0.5,
                    "koi_dror": 0.01,
                    "koi_count": 1,
                    "koi_num_transits": 50
                }
            ]
        }
        
        # Make prediction request
        response = requests.post(
            f"{BASE_URL}/predict",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['predictions'][0]['classification']}")
            print(f"Confidence: {result['predictions'][0]['confidence']:.2%}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
        """, language="python")
        
        st.markdown("""
        ### 3. Understanding the Response
        
        The API returns predictions in this format:
        """)
        
        st.code("""
        {
            "success": true,
            "model_info": {
                "dataset": "kepler",
                "model": "rf",
                "version": "1.0",
                "accuracy": 0.923
            },
            "predictions": [
                {
                    "id": 0,
                    "classification": "CONFIRMED",
                    "confidence": 0.87,
                    "probabilities": {
                        "CONFIRMED": 0.87,
                        "FALSE_POSITIVE": 0.13,
                        "CANDIDATE": 0.00
                    }
                }
            ],
            "processing_time": 0.245
        }
        """, language="json")
    
    def show_authentication_docs(self):
        """Show authentication documentation."""
        st.subheader("🔐 Authentication")
        
        st.markdown("""
        The Exoplanet AI API uses **Bearer Token** authentication. Include your API key 
        in the `Authorization` header of every request.
        """)
        
        st.code("""
        Authorization: Bearer exa_1234567890abcdef1234567890abcdef
        """)
        
        st.markdown("""
        ### API Key Management
        
        - **Security**: Never expose your API key in client-side code
        - **Storage**: Store keys securely using environment variables
        - **Rotation**: Rotate keys regularly for security
        - **Scopes**: Keys have different permission levels
        """)
        
        st.code("""
        # Environment variable approach (recommended)
        import os
        
        API_KEY = os.getenv('EXOPLANET_AI_API_KEY')
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        """, language="python")
        
        st.markdown("""
        ### Authentication Errors
        
        | Status Code | Error | Description |
        |-------------|-------|-------------|
        | 401 | `UNAUTHORIZED` | Missing or invalid API key |
        | 403 | `FORBIDDEN` | API key lacks required permissions |
        | 429 | `RATE_LIMITED` | Too many requests |
        """)
    
    def show_endpoints_reference(self):
        """Show detailed endpoints reference."""
        st.subheader("📋 Endpoints Reference")
        
        for category, endpoints in self.api_endpoints.items():
            st.markdown(f"### {category}")
            
            for endpoint in endpoints:
                with st.expander(f"{endpoint['method']} {endpoint['path']}", expanded=False):
                    st.markdown(f"**{endpoint['description']}**")
                    
                    # Parameters
                    if 'parameters' in endpoint:
                        st.markdown("**Parameters:**")
                        for param in endpoint['parameters']:
                            required = " *(required)*" if param.get('required') else " *(optional)*"
                            st.markdown(f"- `{param['name']}` ({param['type']}){required}: {param['description']}")
                    
                    # Request body
                    if 'request_body' in endpoint:
                        st.markdown("**Request Body:**")
                        st.code(json.dumps(endpoint['request_body'], indent=2), language="json")
                    
                    # Response
                    if 'response' in endpoint:
                        st.markdown("**Response:**")
                        st.code(json.dumps(endpoint['response'], indent=2), language="json")
                    
                    # Example
                    if 'example' in endpoint:
                        st.markdown("**Example:**")
                        st.code(endpoint['example'], language="bash")
    
    def show_data_models(self):
        """Show data models and schemas."""
        st.subheader("📊 Data Models")
        
        models = {
            "PredictionRequest": {
                "description": "Request body for making predictions",
                "properties": {
                    "dataset": {"type": "string", "enum": ["kepler", "tess", "k2"], "description": "Dataset type"},
                    "model": {"type": "string", "enum": ["rf", "logreg", "svm"], "description": "ML model to use"},
                    "candidates": {"type": "array", "description": "Array of candidate objects to analyze"}
                }
            },
            "Candidate": {
                "description": "Exoplanet candidate data",
                "properties": {
                    "koi_period": {"type": "number", "description": "Orbital period in days"},
                    "koi_prad": {"type": "number", "description": "Planet radius in Earth radii"},
                    "koi_depth": {"type": "number", "description": "Transit depth in ppm"},
                    "koi_duration": {"type": "number", "description": "Transit duration in hours"},
                    "koi_ingress": {"type": "number", "description": "Ingress duration in hours"},
                    "koi_dror": {"type": "number", "description": "Planet-star radius ratio"},
                    "koi_count": {"type": "integer", "description": "Number of planets in system"},
                    "koi_num_transits": {"type": "integer", "description": "Number of observed transits"}
                }
            },
            "PredictionResponse": {
                "description": "Response from prediction endpoint",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether request was successful"},
                    "model_info": {"type": "object", "description": "Information about the model used"},
                    "predictions": {"type": "array", "description": "Array of prediction results"},
                    "processing_time": {"type": "number", "description": "Processing time in seconds"}
                }
            },
            "Prediction": {
                "description": "Individual prediction result",
                "properties": {
                    "id": {"type": "integer", "description": "Candidate ID"},
                    "classification": {"type": "string", "enum": ["CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"], "description": "Predicted class"},
                    "confidence": {"type": "number", "description": "Confidence score (0-1)"},
                    "probabilities": {"type": "object", "description": "Class probabilities"}
                }
            }
        }
        
        for model_name, model_data in models.items():
            with st.expander(f"📋 {model_name}", expanded=False):
                st.markdown(f"**{model_data['description']}**")
                
                st.markdown("**Properties:**")
                for prop_name, prop_data in model_data['properties'].items():
                    prop_type = prop_data['type']
                    if 'enum' in prop_data:
                        prop_type += f" (one of: {', '.join(prop_data['enum'])})"
                    
                    st.markdown(f"- `{prop_name}` ({prop_type}): {prop_data['description']}")
    
    def show_code_examples(self):
        """Show code examples in different languages."""
        st.subheader("💻 Code Examples")
        
        language = st.selectbox("Choose a programming language:", ["Python", "JavaScript", "cURL", "R"])
        
        if language == "Python":
            self.show_python_examples()
        elif language == "JavaScript":
            self.show_javascript_examples()
        elif language == "cURL":
            self.show_curl_examples()
        else:
            self.show_r_examples()
    
    def show_python_examples(self):
        """Show Python code examples."""
        st.markdown("### 🐍 Python Examples")
        
        st.markdown("#### Basic Prediction")
        st.code("""
        import requests
        import json
        
        def predict_exoplanets(api_key, candidates_data):
            url = "https://api.exoplanet-ai.com/v1/predict"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "dataset": "kepler",
                "model": "rf",
                "candidates": candidates_data
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        # Example usage
        api_key = "your_api_key_here"
        candidates = [
            {
                "koi_period": 10.5,
                "koi_prad": 1.2,
                "koi_depth": 100,
                "koi_duration": 2.5,
                "koi_ingress": 0.5,
                "koi_dror": 0.01,
                "koi_count": 1,
                "koi_num_transits": 50
            }
        ]
        
        result = predict_exoplanets(api_key, candidates)
        print(json.dumps(result, indent=2))
        """, language="python")
        
        st.markdown("#### Batch Processing")
        st.code("""
        import pandas as pd
        
        def process_csv_file(api_key, csv_file_path):
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            # Convert to list of dictionaries
            candidates = df.to_dict('records')
            
            # Process in batches of 100
            batch_size = 100
            all_results = []
            
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                
                try:
                    result = predict_exoplanets(api_key, batch)
                    all_results.extend(result['predictions'])
                    print(f"Processed batch {i//batch_size + 1}")
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
            
            return all_results
        
        # Process your data
        results = process_csv_file("your_api_key", "exoplanet_candidates.csv")
        """, language="python")
    
    def show_javascript_examples(self):
        """Show JavaScript code examples."""
        st.markdown("### 🟨 JavaScript Examples")
        
        st.code("""
        // Basic prediction function
        async function predictExoplanets(apiKey, candidatesData) {
            const url = 'https://api.exoplanet-ai.com/v1/predict';
            
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    dataset: 'kepler',
                    model: 'rf',
                    candidates: candidatesData
                })
            });
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} - ${await response.text()}`);
            }
            
            return await response.json();
        }
        
        // Example usage
        const apiKey = 'your_api_key_here';
        const candidates = [
            {
                koi_period: 10.5,
                koi_prad: 1.2,
                koi_depth: 100,
                koi_duration: 2.5,
                koi_ingress: 0.5,
                koi_dror: 0.01,
                koi_count: 1,
                koi_num_transits: 50
            }
        ];
        
        predictExoplanets(apiKey, candidates)
            .then(result => {
                console.log('Predictions:', result);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        """, language="javascript")
    
    def show_curl_examples(self):
        """Show cURL examples."""
        st.markdown("### 🌐 cURL Examples")
        
        st.code("""
        # Basic prediction request
        curl -X POST https://api.exoplanet-ai.com/v1/predict \\
          -H "Authorization: Bearer your_api_key_here" \\
          -H "Content-Type: application/json" \\
          -d '{
            "dataset": "kepler",
            "model": "rf",
            "candidates": [
              {
                "koi_period": 10.5,
                "koi_prad": 1.2,
                "koi_depth": 100,
                "koi_duration": 2.5,
                "koi_ingress": 0.5,
                "koi_dror": 0.01,
                "koi_count": 1,
                "koi_num_transits": 50
              }
            ]
          }'
        
        # Get model information
        curl -X GET https://api.exoplanet-ai.com/v1/models \\
          -H "Authorization: Bearer your_api_key_here"
        
        # Get API status
        curl -X GET https://api.exoplanet-ai.com/v1/status \\
          -H "Authorization: Bearer your_api_key_here"
        """, language="bash")
    
    def show_r_examples(self):
        """Show R code examples."""
        st.markdown("### R Examples")
        
        st.code("""
        library(httr)
        library(jsonlite)
        
        # Function to make predictions
        predict_exoplanets <- function(api_key, candidates_data) {
          url <- "https://api.exoplanet-ai.com/v1/predict"
          
          headers <- add_headers(
            "Authorization" = paste("Bearer", api_key),
            "Content-Type" = "application/json"
          )
          
          payload <- list(
            dataset = "kepler",
            model = "rf",
            candidates = candidates_data
          )
          
          response <- POST(url, headers, body = toJSON(payload, auto_unbox = TRUE))
          
          if (status_code(response) == 200) {
            return(fromJSON(content(response, "text")))
          } else {
            stop(paste("API Error:", status_code(response), "-", content(response, "text")))
          }
        }
        
        # Example usage
        api_key <- "your_api_key_here"
        
        candidates <- list(
          list(
            koi_period = 10.5,
            koi_prad = 1.2,
            koi_depth = 100,
            koi_duration = 2.5,
            koi_ingress = 0.5,
            koi_dror = 0.01,
            koi_count = 1,
            koi_num_transits = 50
          )
        )
        
        result <- predict_exoplanets(api_key, candidates)
        print(result)
        """, language="r")
    
    def show_rate_limits(self):
        """Show rate limits and quotas."""
        st.subheader("⏱️ Rate Limits & Quotas")
        
        st.markdown("""
        To ensure fair usage and system stability, the API implements rate limiting:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Requests per minute", "100")
        
        with col2:
            st.metric("Requests per hour", "1,000")
        
        with col3:
            st.metric("Requests per day", "10,000")
        
        st.markdown("""
        ### Rate Limit Headers
        
        Every API response includes rate limit information in the headers:
        """)
        
        st.code("""
        X-RateLimit-Limit: 100
        X-RateLimit-Remaining: 95
        X-RateLimit-Reset: 1640995200
        """)
        
        st.markdown("""
        ### Handling Rate Limits
        
        When you exceed the rate limit, you'll receive a `429 Too Many Requests` response:
        """)
        
        st.code("""
        {
          "error": "rate_limit_exceeded",
          "message": "Rate limit exceeded. Try again in 60 seconds.",
          "retry_after": 60
        }
        """, language="json")
        
        st.markdown("""
        ### Best Practices
        
        - **Implement exponential backoff** when receiving 429 responses
        - **Cache results** when possible to reduce API calls
        - **Use batch requests** to process multiple candidates efficiently
        - **Monitor your usage** through the API dashboard
        """)
    
    def show_error_handling(self):
        """Show error handling documentation."""
        st.subheader("❌ Error Handling")
        
        st.markdown("""
        The API uses standard HTTP status codes and returns detailed error information:
        """)
        
        error_codes = [
            {"code": 200, "status": "OK", "description": "Request successful"},
            {"code": 400, "status": "Bad Request", "description": "Invalid request parameters"},
            {"code": 401, "status": "Unauthorized", "description": "Missing or invalid API key"},
            {"code": 403, "status": "Forbidden", "description": "Insufficient permissions"},
            {"code": 404, "status": "Not Found", "description": "Endpoint not found"},
            {"code": 422, "status": "Unprocessable Entity", "description": "Invalid data format"},
            {"code": 429, "status": "Too Many Requests", "description": "Rate limit exceeded"},
            {"code": 500, "status": "Internal Server Error", "description": "Server error"},
            {"code": 503, "status": "Service Unavailable", "description": "Service temporarily unavailable"}
        ]
        
        for error in error_codes:
            color = "🟢" if error["code"] == 200 else "🔴" if error["code"] >= 500 else "🟡"
            st.markdown(f"{color} **{error['code']} {error['status']}**: {error['description']}")
        
        st.markdown("""
        ### Error Response Format
        
        All error responses follow this format:
        """)
        
        st.code("""
        {
          "error": "error_code",
          "message": "Human-readable error description",
          "details": {
            "field": "Additional error details"
          },
          "request_id": "req_1234567890abcdef"
        }
        """, language="json")
        
        st.markdown("""
        ### Error Handling Example (Python)
        """)
        
        st.code("""
        import requests
        
        def handle_api_response(response):
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 400:
                error_data = response.json()
                raise ValueError(f"Bad request: {error_data['message']}")
            
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            
            elif response.status_code == 429:
                error_data = response.json()
                retry_after = error_data.get('retry_after', 60)
                raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds")
            
            elif response.status_code >= 500:
                raise ServerError("Server error. Please try again later")
            
            else:
                raise APIError(f"Unexpected error: {response.status_code}")
        """, language="python")
    
    def show_sdks_libraries(self):
        """Show SDKs and libraries."""
        st.subheader("📚 SDKs & Libraries")
        
        st.markdown("""
        We provide official SDKs and libraries to make integration easier:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🐍 Python SDK
            
            ```bash
            pip install exoplanet-ai-sdk
            ```
            
            ```python
            from exoplanet_ai import ExoplanetAI
            
            client = ExoplanetAI(api_key="your_key")
            
            result = client.predict(
                dataset="kepler",
                model="rf",
                candidates=[{...}]
            )
            ```
            """)
        
        with col2:
            st.markdown("""
            ### 🟨 JavaScript SDK
            
            ```bash
            npm install exoplanet-ai-js
            ```
            
            ```javascript
            import ExoplanetAI from 'exoplanet-ai-js';
            
            const client = new ExoplanetAI('your_key');
            
            const result = await client.predict({
                dataset: 'kepler',
                model: 'rf',
                candidates: [{...}]
            });
            ```
            """)
        
        st.markdown("""
        ### R Package
        
        ```r
        install.packages("exoplanetai")
        
        library(exoplanetai)
        
        client <- ExoplanetAI$new(api_key = "your_key")
        result <- client$predict(dataset = "kepler", ...)
        ```
        
        ### 🔧 Community Libraries
        
        - **Go**: [exoplanet-ai-go](https://github.com/community/exoplanet-ai-go)
        - **Ruby**: [exoplanet-ai-ruby](https://github.com/community/exoplanet-ai-ruby)
        - **PHP**: [exoplanet-ai-php](https://github.com/community/exoplanet-ai-php)
        - **Java**: [exoplanet-ai-java](https://github.com/community/exoplanet-ai-java)
        """)
    
    def _load_api_endpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load API endpoints data."""
        return {
            "Predictions": [
                {
                    "method": "POST",
                    "path": "/v1/predict",
                    "description": "Make exoplanet predictions for candidate data",
                    "parameters": [
                        {"name": "dataset", "type": "string", "required": True, "description": "Dataset type (kepler, tess, k2)"},
                        {"name": "model", "type": "string", "required": True, "description": "ML model (rf, logreg, svm)"},
                        {"name": "candidates", "type": "array", "required": True, "description": "Array of candidate objects"}
                    ],
                    "example": "curl -X POST https://api.exoplanet-ai.com/v1/predict -H 'Authorization: Bearer API_KEY' -d '{...}'"
                }
            ],
            "Models": [
                {
                    "method": "GET",
                    "path": "/v1/models",
                    "description": "Get information about available models",
                    "example": "curl -X GET https://api.exoplanet-ai.com/v1/models -H 'Authorization: Bearer API_KEY'"
                },
                {
                    "method": "GET",
                    "path": "/v1/models/{model_id}",
                    "description": "Get detailed information about a specific model",
                    "parameters": [
                        {"name": "model_id", "type": "string", "required": True, "description": "Model identifier"}
                    ]
                }
            ],
            "System": [
                {
                    "method": "GET",
                    "path": "/v1/status",
                    "description": "Get API status and health information",
                    "example": "curl -X GET https://api.exoplanet-ai.com/v1/status"
                },
                {
                    "method": "GET",
                    "path": "/v1/usage",
                    "description": "Get your API usage statistics",
                    "example": "curl -X GET https://api.exoplanet-ai.com/v1/usage -H 'Authorization: Bearer API_KEY'"
                }
            ]
        }
    
    def _load_api_examples(self) -> Dict[str, str]:
        """Load API examples."""
        return {
            "basic_prediction": """
            {
                "dataset": "kepler",
                "model": "rf",
                "candidates": [
                    {
                        "koi_period": 10.5,
                        "koi_prad": 1.2,
                        "koi_depth": 100,
                        "koi_duration": 2.5,
                        "koi_ingress": 0.5,
                        "koi_dror": 0.01,
                        "koi_count": 1,
                        "koi_num_transits": 50
                    }
                ]
            }
            """
        }

# Global API documentation instance
api_docs = APIDocumentation()
