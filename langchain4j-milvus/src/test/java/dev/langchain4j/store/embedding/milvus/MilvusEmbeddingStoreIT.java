package dev.langchain4j.store.embedding.milvus;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.milvus.MilvusContainer;

import java.util.Collections;
import java.util.List;

import static dev.langchain4j.internal.Utils.randomUUID;
import static io.milvus.common.clientenum.ConsistencyLevelEnum.STRONG;
import static java.util.Arrays.asList;
import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers
class MilvusEmbeddingStoreIT extends EmbeddingStoreWithFilteringIT {

    private static final String COLLECTION_NAME = "test_collection";
    private static final String PARTITION_NAME = "test_partition";

    @Container
    private static final MilvusContainer milvus = new MilvusContainer("milvusdb/milvus:v2.3.16");

    MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
            .uri(milvus.getEndpoint())
            .collectionName(COLLECTION_NAME)
            .consistencyLevel(STRONG)
            .username(System.getenv("MILVUS_USERNAME"))
            .password(System.getenv("MILVUS_PASSWORD"))
            .dimension(384)
            .retrieveEmbeddingsOnSearch(true)
            .build();

    EmbeddingModel embeddingModel = new AllMiniLmL6V2QuantizedEmbeddingModel();

    @AfterEach
    void afterEach() {
        embeddingStore.dropCollection(COLLECTION_NAME);
    }

    @Override
    protected EmbeddingStore<TextSegment> embeddingStore() {
        return embeddingStore;
    }

    @Override
    protected EmbeddingModel embeddingModel() {
        return embeddingModel;
    }

    @Test
    void should_not_retrieve_embeddings_when_searching() {

        EmbeddingStore<TextSegment> embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .consistencyLevel(STRONG)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();

        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();
        embeddingStore.addAll(asList(firstEmbedding, secondEmbedding));

        List<EmbeddingMatch<TextSegment>> matches = embeddingStore.search(EmbeddingSearchRequest.builder()
                .queryEmbedding(firstEmbedding)
                .maxResults(10)
                .build()).matches();
        assertThat(matches).hasSize(2);
        assertThat(matches.get(0).embedding()).isNull();
        assertThat(matches.get(1).embedding()).isNull();
    }

    @Test
    void should_use_partition_searching() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();

        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();


        embeddingStore.createPartition(PARTITION_NAME);
        embeddingStore.addAll(asList(firstEmbedding, secondEmbedding), PARTITION_NAME);

        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder()
                .queryEmbedding(firstEmbedding)
                .maxResults(10)
                .minScore(0.0)
                .build();

        EmbeddingSearchResult<TextSegment> embeddingSearchResult = embeddingStore.search(embeddingSearchRequest, Collections.singletonList(PARTITION_NAME));
        List<EmbeddingMatch<TextSegment>> relevant = embeddingSearchResult.matches();
        assertThat(relevant).hasSize(2);
        assertThat(relevant.get(0).embedding()).isNull();
        assertThat(relevant.get(1).embedding()).isNull();
    }


    @Test
    void should_use_deleteByIds() {
        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();
        embeddingStore.createPartition(PARTITION_NAME);
        List<String> ids = embeddingStore.addAll(asList(firstEmbedding, secondEmbedding), PARTITION_NAME);
        assertThat(ids).isNotNull();

        embeddingStore.deleteByIds(PARTITION_NAME, ids);

    }

    @Test
    void should_use_deletePartition() {

        String partitionName = PARTITION_NAME + randomUUID().replace("-", "");
        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();

        Embedding firstEmbedding = embeddingModel.embed("hello").content();
        Embedding secondEmbedding = embeddingModel.embed("hi").content();
        embeddingStore.createPartition(partitionName);
        embeddingStore.addAll(asList(firstEmbedding, secondEmbedding));

        embeddingStore.deletePartition(partitionName);
    }

    @Test
    void should_use_releaseCollection() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        embeddingStore.releaseCollection();

    }

    @Test
    void should_use_loadCollection() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        embeddingStore.loadCollection();
    }

    @Test
    void should_use_deleteCollection() {

        MilvusEmbeddingStore embeddingStore = MilvusEmbeddingStore.builder()
                .host(milvus.getHost())
                .port(milvus.getMappedPort(19530))
                .collectionName(COLLECTION_NAME)
                .dimension(384)
                .retrieveEmbeddingsOnSearch(false)
                .build();
        embeddingStore.dropCollection();
    }


}