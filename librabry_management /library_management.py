import sqlite3
import requests

# Database connection
conn = sqlite3.connect('library.db')
cursor = conn.cursor()

# Create tables
def create_tables():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            genre TEXT,
            quantity INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER NOT NULL,
            issued_to TEXT,
            type TEXT NOT NULL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (book_id) REFERENCES books (id)
        )
    ''')
    conn.commit()

# Add a book manually
def add_book(title, author, genre, quantity):
    cursor.execute('''
        INSERT INTO books (title, author, genre, quantity)
        VALUES (?, ?, ?, ?)
    ''', (title, author, genre, quantity))
    conn.commit()
    print("Book added successfully!")

# Add multiple books
def add_multiple_books(books):
    cursor.executemany('''
        INSERT INTO books (title, author, genre, quantity)
        VALUES (?, ?, ?, ?)
    ''', books)
    conn.commit()
    print("Books added successfully!")

# Fetch book details from Open Library API
def fetch_book_details(isbn):
    base_url = "https://openlibrary.org/api/books"
    params = {
        "bibkeys": f"ISBN:{isbn}",
        "format": "json",
        "jscmd": "data"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        key = f"ISBN:{isbn}"
        if key in data:
            book = data[key]
            title = book.get("title", "N/A")
            authors = [author["name"] for author in book.get("authors", [])]
            return {"title": title, "author": ', '.join(authors)}
        else:
            print("No book data found for the provided ISBN.")
            return None
    else:
        print("Failed to fetch data from Open Library.")
        return None

# Add a book via API
def add_book_via_api(isbn, genre, quantity):
    book_details = fetch_book_details(isbn)
    if book_details:
        add_book(book_details["title"], book_details["author"], genre, quantity)

# Add sample books
def add_sample_books():
    sample_books = [
        ("To Kill a Mockingbird", "Harper Lee", "Fiction", 5),
        ("1984", "George Orwell", "Dystopian", 8),
        ("The Great Gatsby", "F. Scott Fitzgerald", "Classic", 3),
        ("Pride and Prejudice", "Jane Austen", "Romance", 6),
        ("The Catcher in the Rye", "J.D. Salinger", "Classic", 4),
        ("Moby-Dick", "Herman Melville", "Adventure", 2),
        ("War and Peace", "Leo Tolstoy", "Historical", 5),
    ]
    add_multiple_books(sample_books)

# Issue a book
def issue_book(book_id, issued_to):
    cursor.execute('SELECT quantity FROM books WHERE id = ?', (book_id,))
    result = cursor.fetchone()
    if result and result[0] > 0:
        cursor.execute('UPDATE books SET quantity = quantity - 1 WHERE id = ?', (book_id,))
        cursor.execute('''
            INSERT INTO transactions (book_id, issued_to, type)
            VALUES (?, ?, 'issue')
        ''', (book_id, issued_to))
        conn.commit()
        print("Book issued successfully!")
    else:
        print("Book is not available.")

# Return a book
def return_book(book_id, issued_to):
    cursor.execute('SELECT id FROM transactions WHERE book_id = ? AND issued_to = ? AND type = "issue"', 
                   (book_id, issued_to))
    result = cursor.fetchone()
    if result:
        cursor.execute('UPDATE books SET quantity = quantity + 1 WHERE id = ?', (book_id,))
        cursor.execute('''
            INSERT INTO transactions (book_id, issued_to, type)
            VALUES (?, ?, 'return')
        ''', (book_id, issued_to))
        conn.commit()
        print("Book returned successfully!")
    else:
        print("No record of this book being issued to the user.")

# Check book availability
def check_availability(book_id):
    cursor.execute('SELECT title, author, quantity FROM books WHERE id = ?', (book_id,))
    book = cursor.fetchone()
    if book:
        title, author, quantity = book
        print(f"Title: {title}, Author: {author}, Available Copies: {quantity}")
    else:
        print("Book not found.")

# Search books by title
def search_books_by_title(title):
    cursor.execute('SELECT id, title, author, quantity FROM books WHERE title LIKE ?', (f"%{title}%",))
    books = cursor.fetchall()
    for book in books:
        print(f"ID: {book[0]}, Title: {book[1]}, Author: {book[2]}, Quantity: {book[3]}")

# Main menu
def main():
    create_tables()
    while True:
        print("\nLibrary Management System")
        print("1. Add a book manually")
        print("2. Add a book via API")
        print("3. Issue a book")
        print("4. Return a book")
        print("5. Check book availability")
        print("6. Search books by title")
        print("7. Add sample books")
        print("8. Add multiple books interactively")
        print("9. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            title = input("Enter title: ")
            author = input("Enter author: ")
            genre = input("Enter genre: ")
            quantity = int(input("Enter quantity: "))
            add_book(title, author, genre, quantity)
        elif choice == "2":
            isbn = input("Enter ISBN: ")
            genre = input("Enter genre: ")
            quantity = int(input("Enter quantity: "))
            add_book_via_api(isbn, genre, quantity)
        elif choice == "3":
            book_id = int(input("Enter book ID: "))
            issued_to = input("Enter the name of the person: ")
            issue_book(book_id, issued_to)
        elif choice == "4":
            book_id = int(input("Enter book ID: "))
            issued_to = input("Enter the name of the person: ")
            return_book(book_id, issued_to)
        elif choice == "5":
            book_id = int(input("Enter book ID: "))
            check_availability(book_id)
        elif choice == "6":
            title = input("Enter title to search: ")
            search_books_by_title(title)
        elif choice == "7":
            add_sample_books()
        elif choice == "8":
            print("Enter book details (or type 'done' to finish):")
            books = []
            while True:
                title = input("Enter title: ")
                if title.lower() == 'done':
                    break
                author = input("Enter author: ")
                genre = input("Enter genre: ")
                quantity = int(input("Enter quantity: "))
                books.append((title, author, genre, quantity))
            add_multiple_books(books)
        elif choice == "9":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the application
if __name__ == "__main__":
    main()
