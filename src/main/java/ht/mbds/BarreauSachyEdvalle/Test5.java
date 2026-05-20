package ht.mbds.BarreauSachyEdvalle;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.util.List;
import java.util.Scanner;

public class Test5 {

    public static void main(String[] args) {

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);

        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(System.getenv("TAVILY_API_KEY"))
                        .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        QueryRouter queryRouter =
                new DefaultQueryRouter(
                        pdfRetriever,
                        webRetriever
                );

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();


        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.println("Posez votre question : ");

                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }



                String reponse = assistant.chat(question);

                System.out.println("Assistant : " + reponse);

            }
        }
    }
}