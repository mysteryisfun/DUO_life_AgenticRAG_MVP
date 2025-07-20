### **API Endpoints & JSON Payloads**

-----

#### **1. Generate New Session**

  * **Endpoint:** `GET /new_session`
  * **Request Body:**
      * None
  * **Response Body (200 OK):**
    ```json
    {
      "session_id": "a1b2-c3d4-e5f6-g7h8"
    }
    ```

-----

#### **2. Query the Agent (Streaming)**

  * **Endpoint:** `POST /query`

  * **Request Body:**

    ```json
    {
      "question": "Are there any liquid products for skin health?",
      "session_id": "a1b2-c3d4-e5f6-g7h8"
    }
    ```

  * **Response Body (Streaming):**
    *A sequence of JSON objects sent over the connection.*

      * **Object Type 1: Status Log**
        ```json
        {
          "type": "log",
          "data": {
            "message": "Searching knowledge base..."
          }
        }
        ```
      * **Object Type 2: Answer Token**
        ```json
        {
          "type": "token",
          "data": {
            "chunk": "Yes, DuoLife Collagen "
          }
        }
        ```
      * **Object Type 3: Final Response**
        ```json
        {
          "type": "final_response",
          "data": {
            "session_id": "a1b2-c3d4-e5f6-g7h8",
            "final_answer": "Yes, DuoLife Collagen is a liquid supplement designed to support skin, bones, and joints.",
            "sources": [
              {
                "name": "DuoLife Collagen",
                "links": [
                  "https://myduolife.com/shop/products/1/317,0,duolife-collagen.html"
                ],
                "type": "Product",
                "category": "Diatary Suplement",
                "snippet": "100% natural dietary supplement... supporting the skin, bones and joints."
              }
            ]
          }
        }
        ```

-----

#### **3. Clear Session Memory**

  * **Endpoint:** `POST /clear_memory`
  * **Request Body:**
    ```json
    {
      "session_id": "a1b2-c3d4-e5f6-g7h8"
    }
    ```
  * **Response Body (200 OK):**
    ```json
    {
      "message": "Conversation memory for session a1b2-c3d4-e5f6-g7h8 has been cleared."
    }
    ```