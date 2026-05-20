package ht.mbds.BarreauSachyEdvalle;

import java.util.Scanner;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.parser.TextDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.List;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import java.util.Map;
import java.util.HashMap;

public class TestRoutage {

    public static void main(String[] args) {

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash-lite")
                .logRequestsAndResponses(true)
                .build();

        System.out.println("ChatModel créé");
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> storeIA =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        EmbeddingStore<TextSegment> storeCuisine =
                creerEmbeddingStore("HadoopSparkMapReduce_1.pdf", embeddingModel);
        ContentRetriever retrieverIA =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeIA)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        ContentRetriever retrieverHadoop =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(storeCuisine)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        System.out.println("ContentRetrievers créés");
        Map<ContentRetriever, String> retrievers = new HashMap<>();

        retrievers.put(
                retrieverIA,
                "Documents sur l'intelligence artificielle, le RAG et le fine-tuning"
        );

        retrievers.put(
                retrieverHadoop,
                "Documents sur Hadoop, Spark et MapReduce"
        );

        QueryRouter queryRouter =
                new LanguageModelQueryRouter(model, retrievers);

        System.out.println("QueryRouter créé");
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        System.out.println("RetrievalAugmentor créé");
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant créé");
        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.println("==========================");
                System.out.println("Posez votre question : ");

                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }

                System.out.println("===================");

                String reponse = assistant.chat(question);

                System.out.println("Assistant : " + reponse);

            }
        }
    }
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(
            String nomFichier,
            EmbeddingModel embeddingModel
    ) {
        Document document;

        if (nomFichier.endsWith(".pdf")) {
            document = ClassPathDocumentLoader.loadDocument(
                    nomFichier,
                    new ApacheTikaDocumentParser()
            );
        } else {
            document = ClassPathDocumentLoader.loadDocument(
                    nomFichier,
                    new TextDocumentParser()
            );
        }

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        Response<List<Embedding>> response = embeddingModel.embedAll(segments);

        List<Embedding> embeddings = response.content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Document chargé : " + nomFichier);
        System.out.println("Segments : " + segments.size());

        return embeddingStore;
    }
}